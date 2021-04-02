#include <cassert>
#include <Common/Exception.h>

#include <DataStreams/IBlockInputStream.h>

#include <Interpreters/MutationsInterpreter.h>
#include <Storages/StorageFactory.h>
#include <Storages/StorageAggregatingMemory.h>

#include <IO/WriteHelpers.h>
#include <Processors/Sources/SourceWithProgress.h>
#include <Processors/Pipe.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int INCORRECT_QUERY;
}


class MemorySource : public SourceWithProgress
{
    using InitializerFunc = std::function<void(std::shared_ptr<const Blocks> &)>;
public:
    /// Blocks are stored in std::list which may be appended in another thread.
    /// We use pointer to the beginning of the list and its current size.
    /// We don't need synchronisation in this reader, because while we hold SharedLock on storage,
    /// only new elements can be added to the back of the list, so our iterators remain valid

    MemorySource(
        Names column_names_,
        const StorageAggregatingMemory & storage,
        const StorageMetadataPtr & metadata_snapshot,
        std::shared_ptr<const Blocks> data_,
        std::shared_ptr<std::atomic<size_t>> parallel_execution_index_,
        InitializerFunc initializer_func_ = {})
        : SourceWithProgress(metadata_snapshot->getSampleBlockForColumns(column_names_, storage.getVirtuals(), storage.getStorageID()))
        , column_names_and_types(metadata_snapshot->getColumns().getAllWithSubcolumns().addTypes(std::move(column_names_)))
        , data(data_)
        , parallel_execution_index(parallel_execution_index_)
        , initializer_func(std::move(initializer_func_))
    {
    }

    String getName() const override { return "AggregatingMemory"; }

protected:
    Chunk generate() override
    {
        if (initializer_func)
        {
            initializer_func(data);
            initializer_func = {};
        }

        size_t current_index = getAndIncrementExecutionIndex();

        if (current_index >= data->size())
        {
            return {};
        }

        const Block & src = (*data)[current_index];
        Columns columns;
        columns.reserve(columns.size());

        /// Add only required columns to `res`.
        for (const auto & elem : column_names_and_types)
        {
            auto current_column = src.getByName(elem.getNameInStorage()).column;
            if (elem.isSubcolumn())
                columns.emplace_back(elem.getTypeInStorage()->getSubcolumn(elem.getSubcolumnName(), *current_column));
            else
                columns.emplace_back(std::move(current_column));
        }

        return Chunk(std::move(columns), src.rows());
    }

private:
    size_t getAndIncrementExecutionIndex()
    {
        if (parallel_execution_index)
        {
            return (*parallel_execution_index)++;
        }
        else
        {
            return execution_index++;
        }
    }

    const NamesAndTypesList column_names_and_types;
    size_t execution_index = 0;
    std::shared_ptr<const Blocks> data;
    std::shared_ptr<std::atomic<size_t>> parallel_execution_index;
    InitializerFunc initializer_func;
};


class AggregatingOutputStream : public IBlockOutputStream
{
public:
    AggregatingOutputStream(
        StorageAggregatingMemory & storage_,
        const StorageMetadataPtr & metadata_snapshot_)
        : storage(storage_)
        , metadata_snapshot(metadata_snapshot_)
    {
    }

    Block getHeader() const override { return metadata_snapshot->getSampleBlock(); }

    void write(const Block & block) override
    {
        metadata_snapshot->check(block, true);
        new_blocks.emplace_back(block);
    }

    void writeSuffix() override
    {
        size_t inserted_bytes = 0;
        size_t inserted_rows = 0;

        for (const auto & block : new_blocks)
        {
            inserted_bytes += block.allocatedBytes();
            inserted_rows += block.rows();
        }

        std::lock_guard lock(storage.mutex);

        auto new_data = std::make_unique<Blocks>(*(storage.data.get()));
        new_data->insert(new_data->end(), new_blocks.begin(), new_blocks.end());

        storage.data.set(std::move(new_data));
        storage.total_size_bytes.fetch_add(inserted_bytes, std::memory_order_relaxed);
        storage.total_size_rows.fetch_add(inserted_rows, std::memory_order_relaxed);
    }

private:
    Blocks new_blocks;

    StorageAggregatingMemory & storage;
    StorageMetadataPtr metadata_snapshot;
};


StorageAggregatingMemory::StorageAggregatingMemory(const StorageID & table_id_, ColumnsDescription columns_description_, ConstraintsDescription constraints_, const ASTCreateQuery & query)
    : IStorage(table_id_), data(std::make_unique<const Blocks>())
{
    // TODO: this table must be created with original write structure, and aggregated read structure
    // TODO: also i should add metadata to indicate that aggregation is not needed in this case.

    LOG_DEBUG(&Poco::Logger::get("Arthur"), "create engine with query={}", serializeAST(query));

    if (!query.select)
        throw Exception("SELECT query is not specified for " + getName(), ErrorCodes::INCORRECT_QUERY);

    if (query.select->list_of_selects->children.size() != 1)
        throw Exception("UNION is not supported for AggregatingMemory", ErrorCodes::INCORRECT_QUERY);

    StorageInMemoryMetadata storage_metadata;
    storage_metadata.setColumns(std::move(columns_description_));
    storage_metadata.setConstraints(std::move(constraints_));

    // TODO: check GROUP BY inside this func
    auto select = SelectQueryDescription::getSelectQueryFromASTForAggr(query.select->clone());
    storage_metadata.setSelectQuery(select);

    setInMemoryMetadata(storage_metadata);
}


Pipe StorageAggregatingMemory::read(
    const Names & column_names,
    const StorageMetadataPtr & metadata_snapshot,
    SelectQueryInfo & /*query_info*/,
    const Context & /*context*/,
    QueryProcessingStage::Enum /*processed_stage*/,
    size_t /*max_block_size*/,
    unsigned num_streams)
{
    metadata_snapshot->check(column_names, getVirtuals(), getStorageID());

    auto current_data = data.get();
    size_t size = current_data->size();

    if (num_streams > size)
        num_streams = size;

    Pipes pipes;

    auto parallel_execution_index = std::make_shared<std::atomic<size_t>>(0);

    for (size_t stream = 0; stream < num_streams; ++stream)
    {
        pipes.emplace_back(std::make_shared<MemorySource>(column_names, *this, metadata_snapshot, current_data, parallel_execution_index));
    }

    return Pipe::unitePipes(std::move(pipes));
}


BlockOutputStreamPtr StorageAggregatingMemory::write(const ASTPtr & /*query*/, const StorageMetadataPtr & metadata_snapshot, const Context & /*context*/)
{
    return std::make_shared<AggregatingOutputStream>(*this, metadata_snapshot);
}


void StorageAggregatingMemory::drop()
{
    data.set(std::make_unique<Blocks>());
    total_size_bytes.store(0, std::memory_order_relaxed);
    total_size_rows.store(0, std::memory_order_relaxed);
}

static inline void updateBlockData(Block & old_block, const Block & new_block)
{
    for (const auto & it : new_block)
    {
        auto col_name = it.name;
        auto & col_with_type_name = old_block.getByName(col_name);
        col_with_type_name.column = it.column;
    }
}

void StorageAggregatingMemory::mutate(const MutationCommands & commands, const Context & context)
{
    std::lock_guard lock(mutex);
    auto metadata_snapshot = getInMemoryMetadataPtr();
    auto storage = getStorageID();
    auto storage_ptr = DatabaseCatalog::instance().getTable(storage, context);
    auto interpreter = std::make_unique<MutationsInterpreter>(storage_ptr, metadata_snapshot, commands, context, true);
    auto in = interpreter->execute();

    in->readPrefix();
    Blocks out;
    Block block;
    while ((block = in->read()))
    {
        out.push_back(block);
    }
    in->readSuffix();

    std::unique_ptr<Blocks> new_data;

    // all column affected
    if (interpreter->isAffectingAllColumns())
    {
        new_data = std::make_unique<Blocks>(out);
    }
    else
    {
        /// just some of the column affected, we need update it with new column
        new_data = std::make_unique<Blocks>(*(data.get()));
        auto data_it = new_data->begin();
        auto out_it = out.begin();

        while (data_it != new_data->end())
        {
            /// Mutation does not change the number of blocks
            assert(out_it != out.end());

            updateBlockData(*data_it, *out_it);
            ++data_it;
            ++out_it;
        }

        assert(out_it == out.end());
    }

    size_t rows = 0;
    size_t bytes = 0;
    for (const auto & buffer : *new_data)
    {
        rows += buffer.rows();
        bytes += buffer.bytes();
    }
    total_size_bytes.store(rows, std::memory_order_relaxed);
    total_size_rows.store(bytes, std::memory_order_relaxed);
    data.set(std::move(new_data));
}


void StorageAggregatingMemory::truncate(
    const ASTPtr &, const StorageMetadataPtr &, const Context &, TableExclusiveLockHolder &)
{
    data.set(std::make_unique<Blocks>());
    total_size_bytes.store(0, std::memory_order_relaxed);
    total_size_rows.store(0, std::memory_order_relaxed);
}

std::optional<UInt64> StorageAggregatingMemory::totalRows(const Settings &) const
{
    /// All modifications of these counters are done under mutex which automatically guarantees synchronization/consistency
    /// When run concurrently we are fine with any value: "before" or "after"
    return total_size_rows.load(std::memory_order_relaxed);
}

std::optional<UInt64> StorageAggregatingMemory::totalBytes(const Settings &) const
{
    return total_size_bytes.load(std::memory_order_relaxed);
}

void registerStorageAggregatingMemory(StorageFactory & factory)
{
    factory.registerStorage("AggregatingMemory", [](const StorageFactory::Arguments & args)
    {
        if (!args.engine_args.empty())
            throw Exception(
                "Engine " + args.engine_name + " doesn't support any arguments (" + toString(args.engine_args.size()) + " given)",
                ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

        return StorageAggregatingMemory::create(args.table_id, args.columns, args.constraints, args.query);
    },
    {
        .supports_parallel_insert = true,
    });
}

}
