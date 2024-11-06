#include <Processors/QueryPlan/JoinStepLogical.h>

#include <Processors/QueryPlan/JoinStep.h>

#include <QueryPipeline/QueryPipelineBuilder.h>
#include <Processors/Transforms/JoiningTransform.h>
#include <Interpreters/IJoin.h>
#include <Interpreters/TableJoin.h>
#include <Interpreters/Context.h>
#include <IO/Operators.h>
#include <Common/JSONBuilder.h>
#include <Common/typeid_cast.h>
#include <Interpreters/TableJoin.h>
#include <Interpreters/HashJoin/HashJoin.h>
#include <Storages/StorageJoin.h>
#include <ranges>
#include <Core/Settings.h>
#include <Functions/FunctionFactory.h>
#include <Interpreters/PasteJoin.h>

namespace DB
{

namespace Setting
{
    extern const SettingsJoinAlgorithm join_algorithm;
    extern const SettingsBool join_any_take_last_row;
}

namespace ErrorCodes
{
    extern const int NOT_IMPLEMENTED;
    extern const int LOGICAL_ERROR;
    extern const int INVALID_JOIN_ON_EXPRESSION;
}

std::string_view toString(PredicateOperator op)
{
    switch (op)
    {
        case PredicateOperator::Equal: return "=";
        case PredicateOperator::NullSafeEqual: return "<=>";
        case PredicateOperator::Less: return "<";
        case PredicateOperator::LessOrEquals: return "<=";
        case PredicateOperator::Greater: return ">";
        case PredicateOperator::GreaterOrEquals: return ">=";
    }
    throw Exception(ErrorCodes::LOGICAL_ERROR, "Illegal value for PredicateOperator: {}", static_cast<Int32>(op));
}


std::string toFunctionName(PredicateOperator op)
{
    switch (op)
    {
        case PredicateOperator::Equal: return "equals";
        case PredicateOperator::NullSafeEqual: return "isNotDistinctFrom";
        case PredicateOperator::Less: return "less";
        case PredicateOperator::LessOrEquals: return "lessOrEquals";
        case PredicateOperator::Greater: return "greater";
        case PredicateOperator::GreaterOrEquals: return "greaterOrEquals";
    }
    throw Exception(ErrorCodes::LOGICAL_ERROR, "Illegal value for PredicateOperator: {}", static_cast<Int32>(op));
}

std::optional<ASOFJoinInequality> operatorToAsofInequality(PredicateOperator op)
{
    switch (op)
    {
        case PredicateOperator::Less: return ASOFJoinInequality::Less;
        case PredicateOperator::LessOrEquals: return ASOFJoinInequality::LessOrEquals;
        case PredicateOperator::Greater: return ASOFJoinInequality::Greater;
        case PredicateOperator::GreaterOrEquals: return ASOFJoinInequality::GreaterOrEquals;
        default: return {};
    }
}

void formatJoinCondition(const JoinCondition & join_condition, WriteBuffer & buf)
{
    auto quote_string = std::views::transform([](const auto & s) { return fmt::format("({})", s.column_name); });
    auto format_predicate = std::views::transform([](const auto & p) { return fmt::format("{} {} {}", p.left_node.column_name, toString(p.op), p.right_node.column_name); });
    buf << "[";
    buf << fmt::format("Keys: ({})", fmt::join(join_condition.predicates | format_predicate, " AND "));
    if (!join_condition.left_filter_conditions.empty())
        buf << " " << fmt::format("Left: ({})", fmt::join(join_condition.left_filter_conditions | quote_string, " AND "));
    if (!join_condition.right_filter_conditions.empty())
        buf << " " << fmt::format("Right: ({})", fmt::join(join_condition.right_filter_conditions | quote_string, " AND "));
    if (!join_condition.residual_conditions.empty())
        buf << " " << fmt::format("Residual: ({})", fmt::join(join_condition.residual_conditions | quote_string, " AND "));
    buf << "]";
}

std::vector<std::pair<String, String>> describeJoinActions(const JoinInfo & join_info)
{
    std::vector<std::pair<String, String>> description;

    description.emplace_back("Type", toString(join_info.kind));
    description.emplace_back("Strictness", toString(join_info.strictness));
    description.emplace_back("Locality", toString(join_info.locality));

    {
        WriteBufferFromOwnString join_expression_str;
        join_expression_str << (join_info.expression.is_using ? "USING" : "ON") << " " ;
        formatJoinCondition(join_info.expression.condition, join_expression_str);
        for (const auto & condition : join_info.expression.disjunctive_conditions)
        {
            join_expression_str << " | ";
            formatJoinCondition(condition, join_expression_str);
        }
        description.emplace_back("Expression", join_expression_str.str());
    }

    return description;
}


JoinStepLogical::JoinStepLogical(
    const Block & left_header_,
    const Block & right_header_,
    JoinInfo join_info_,
    JoinExpressionActions join_expression_actions_,
    Names required_output_columns_,
    ContextPtr context_)
    : expression_actions(std::move(join_expression_actions_))
    , join_info(std::move(join_info_))
    , required_output_columns(std::move(required_output_columns_))
    , query_context(std::move(context_))
{
    updateInputHeaders({left_header_, right_header_});
}

QueryPipelineBuilderPtr JoinStepLogical::updatePipeline(QueryPipelineBuilders /* pipelines */, const BuildQueryPipelineSettings & /* settings */)
{
    throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Cannot execute JoinStepLogical, it should be converted physical step first");
}


void JoinStepLogical::describePipeline(FormatSettings & settings) const
{
    IQueryPlanStep::describePipeline(processors, settings);
}

void JoinStepLogical::describeActions(FormatSettings & settings) const
{
    String prefix(settings.offset, settings.indent_char);
    String prefix2(settings.offset + settings.indent, settings.indent_char);

    for (const auto & [name, value] : describeJoinActions(join_info))
        settings.out << prefix << name << ": " << value << '\n';
    settings.out << prefix << "Post Expression:\n";
    ExpressionActions(expression_actions.post_join_actions.clone()).describeActions(settings.out, prefix2);
    settings.out << prefix << "Left Expression:\n";
    // settings.out << expression_actions.left_pre_join_actions.dumpDAG();
    ExpressionActions(expression_actions.left_pre_join_actions.clone()).describeActions(settings.out, prefix2);
    settings.out << prefix << "Right Expression:\n";
    ExpressionActions(expression_actions.right_pre_join_actions.clone()).describeActions(settings.out, prefix2);
}

void JoinStepLogical::describeActions(JSONBuilder::JSONMap & map) const
{
    for (const auto & [name, value] : describeJoinActions(join_info))
        map.add(name, value);

    map.add("Left Actions", ExpressionActions(expression_actions.left_pre_join_actions.clone()).toTree());
    map.add("Right Actions", ExpressionActions(expression_actions.right_pre_join_actions.clone()).toTree());
    map.add("Post Actions", ExpressionActions(expression_actions.post_join_actions.clone()).toTree());
}

static Block stackHeadersFromStreams(const Headers & input_headers, const Names & required_output_columns)
{
    NameSet required_output_columns_set(required_output_columns.begin(), required_output_columns.end());

    Block result_header;
    for (const auto & header : input_headers)
    {
        for (const auto & column : header)
        {
            if (required_output_columns_set.contains(column.name))
            {
                result_header.insert(column);
            }
            else if (required_output_columns_set.empty())
            {
                /// If no required columns specified, use one first column.
                result_header.insert(column);
                return result_header;
            }
        }
    }
    return result_header;
}

void JoinStepLogical::updateOutputHeader()
{
    output_header = stackHeadersFromStreams(input_headers, required_output_columns);
}


JoinActionRef concatConditions(const std::vector<JoinActionRef> & conditions, ActionsDAG & actions_dag, const ContextPtr & query_context)
{
    if (conditions.empty())
        return JoinActionRef(nullptr);

    if (conditions.size() == 1)
    {
        actions_dag.addOrReplaceInOutputs(*conditions.front().node);
        return conditions.front();
    }

    auto and_function = FunctionFactory::instance().get("and", query_context);
    ActionsDAG::NodeRawConstPtrs nodes;
    nodes.reserve(conditions.size());
    for (const auto & condition : conditions)
        nodes.push_back(condition.node);

    const auto & result_node = actions_dag.addFunction(and_function, nodes, {});
    actions_dag.addOrReplaceInOutputs(result_node);
    return conditions.front();
}

JoinActionRef concatMergeConditions(std::vector<JoinActionRef> & conditions, ActionsDAG & actions_dag, const ContextPtr & query_context)
{
    auto condition = concatConditions(conditions, actions_dag, query_context);
    conditions.clear();
    if (condition)
        conditions = {condition};
    return condition;
}



/// Can be used when action.node is outside of actions_dag.
const ActionsDAG::Node & addInputIfAbsent(ActionsDAG & actions_dag, const JoinActionRef & action)
{
    for (const auto * node : actions_dag.getInputs())
    {
        if (node->result_name == action.column_name)
        {
            if (!node->result_type->equals(*action.node->result_type))
                throw Exception(ErrorCodes::LOGICAL_ERROR, "Column '{}' expected to have type {} but got {}, in actions DAG: {}",
                    action.column_name, action.node->result_type->getName(), node->result_type->getName(), actions_dag.dumpDAG());
            return *node;
        }
    }
    return actions_dag.addInput(action.column_name, action.node->result_type);
}


JoinActionRef predicateToCondition(const JoinPredicate & predicate, ActionsDAG & actions_dag, const ContextPtr & query_context)
{
    const auto & left_node = addInputIfAbsent(actions_dag, predicate.left_node);
    const auto & right_node = addInputIfAbsent(actions_dag, predicate.right_node);

    auto operator_function = FunctionFactory::instance().get(toFunctionName(predicate.op), query_context);
    const auto & result_node = actions_dag.addFunction(operator_function, {&left_node, &right_node}, {});
    return JoinActionRef(&result_node);
}

bool canPushDownFromOn(const JoinInfo & join_info, std::optional<JoinTableSide> side = {})
{
    if (!join_info.expression.disjunctive_conditions.empty())
        return false;

    if (join_info.strictness != JoinStrictness::All
     && join_info.strictness != JoinStrictness::Any
     && join_info.strictness != JoinStrictness::RightAny
     && join_info.strictness != JoinStrictness::Semi)
        return false;

    return join_info.kind == JoinKind::Inner
        || join_info.kind == JoinKind::Cross
        || join_info.kind == JoinKind::Comma
        || join_info.kind == JoinKind::Paste
        || (side == JoinTableSide::Left && join_info.kind == JoinKind::Right)
        || (side == JoinTableSide::Right && join_info.kind == JoinKind::Left);
}

static void addRequiredInputToOutput(ActionsDAG & dag, const NameSet & required_output_columns)
{
    NameSet existing_output_columns;
    for (const auto & node : dag.getOutputs())
        existing_output_columns.insert(node->result_name);

    for (const auto * node : dag.getInputs())
    {
        if (!required_output_columns.contains(node->result_name)
         || existing_output_columns.contains(node->result_name))
            continue;
        dag.addOrReplaceInOutputs(*node);
    }
}

void addJoinConditionToTableJoin(JoinCondition & join_condition, TableJoin::JoinOnClause & table_join_clause, ActionsDAG * post_join_actions, const ContextPtr & query_context)
{
    std::vector<JoinPredicate> new_predicates;
    for (size_t i = 0; i < join_condition.predicates.size(); ++i)
    {
        const auto & predicate = join_condition.predicates[i];
        if (PredicateOperator::Equal == predicate.op || PredicateOperator::NullSafeEqual == predicate.op)
        {
            table_join_clause.addKey(predicate.left_node.column_name, predicate.right_node.column_name, PredicateOperator::NullSafeEqual == predicate.op);
            new_predicates.push_back(predicate);
        }
        else if (post_join_actions)
        {
            auto predicate_action = predicateToCondition(predicate, *post_join_actions, query_context);
            join_condition.residual_conditions.push_back(predicate_action);
        }
    }
    join_condition.predicates = std::move(new_predicates);
}


void addRequiredOutputs(ActionsDAG & actions_dag, const Names & required_output_columns)
{
    NameSet required_output_columns_set(required_output_columns.begin(), required_output_columns.end());
    for (const auto * node : actions_dag.getInputs())
    {
        if (required_output_columns_set.contains(node->result_name))
            actions_dag.addOrReplaceInOutputs(*node);
    }
}


JoinActionRef buildSingleActionForJoinExpression(const JoinCondition & join_condition, JoinExpressionActions & expression_actions, const ContextPtr & query_context)
{
    std::vector<JoinActionRef> all_conditions;
    auto left_filter_conditions_action = concatConditions(join_condition.left_filter_conditions, expression_actions.left_pre_join_actions, query_context);
    if (left_filter_conditions_action)
    {
        left_filter_conditions_action.node = &addInputIfAbsent(expression_actions.post_join_actions, left_filter_conditions_action);
        all_conditions.push_back(left_filter_conditions_action);
    }

    auto right_filter_conditions_action = concatConditions(join_condition.right_filter_conditions, expression_actions.right_pre_join_actions, query_context);
    if (right_filter_conditions_action)
    {
        right_filter_conditions_action.node = &addInputIfAbsent(expression_actions.post_join_actions, right_filter_conditions_action);
        all_conditions.push_back(right_filter_conditions_action);
    }

    for (const auto & predicate : join_condition.predicates)
    {
        auto predicate_action = predicateToCondition(predicate, expression_actions.post_join_actions, query_context);
        all_conditions.push_back(predicate_action);
    }

    return concatConditions(all_conditions, expression_actions.post_join_actions, query_context);
}

JoinActionRef buildSingleActionForJoinExpression(const JoinExpression & join_expression, JoinExpressionActions & expression_actions, const ContextPtr & query_context)
{
    std::vector<JoinActionRef> all_conditions;
    all_conditions.push_back(buildSingleActionForJoinExpression(join_expression.condition, expression_actions, query_context));
    for (const auto & join_condition : join_expression.disjunctive_conditions)
        all_conditions.push_back(buildSingleActionForJoinExpression(join_condition, expression_actions, query_context));
    return concatConditions(all_conditions, expression_actions.post_join_actions, query_context);
}

JoinPtr JoinStepLogical::chooseJoinAlgorithm(JoinActionRef & left_filter, JoinActionRef & right_filter, JoinActionRef & post_filter, bool is_explain_logical)
{
    const auto & settings = query_context->getSettingsRef();

    auto table_join = std::make_shared<TableJoin>(settings, query_context->getGlobalTemporaryVolume(), query_context->getTempDataOnDisk());
    table_join->setJoinInfo(join_info);

    auto & join_expression = join_info.expression;

    std::visit([&](auto && storage_)
    {
        if (storage_ && join_expression.disjunctive_conditions.empty())
            table_join->setStorageJoin(storage_);
    }, prepared_join_storage);

    auto & table_join_clauses = table_join->getClauses();
    addJoinConditionToTableJoin(
        join_expression.condition, table_join_clauses.emplace_back(),
        join_info.strictness != JoinStrictness::Asof ? &expression_actions.post_join_actions : nullptr,
        query_context);

    if (auto left_pre_filter_condition = concatMergeConditions(join_expression.condition.left_filter_conditions, expression_actions.left_pre_join_actions, query_context))
    {
        if (canPushDownFromOn(join_info, JoinTableSide::Left))
            left_filter = left_pre_filter_condition;
        else
            table_join_clauses.back().analyzer_left_filter_condition_column_name = left_pre_filter_condition.column_name;
    }

    if (auto right_pre_filter_condition = concatMergeConditions(join_expression.condition.right_filter_conditions, expression_actions.right_pre_join_actions, query_context))
    {
        if (canPushDownFromOn(join_info, JoinTableSide::Right))
            right_filter = right_pre_filter_condition;
        else
            table_join_clauses.back().analyzer_right_filter_condition_column_name = right_pre_filter_condition.column_name;
    }

    if (join_info.strictness == JoinStrictness::Asof)
    {
        if (!join_info.expression.disjunctive_conditions.empty())
            throw Exception(ErrorCodes::INVALID_JOIN_ON_EXPRESSION, "ASOF join does not support multiple disjuncts in JOIN ON expression");

        /// Find strictly only one inequality in predicate list for ASOF join
        chassert(table_join_clauses.size() == 1);
        const auto & join_predicates = join_info.expression.condition.predicates;
        bool asof_predicate_found = false;
        for (size_t i = 0; i < join_predicates.size(); ++i)
        {
            const auto & predicate = join_predicates[i];
            auto asof_inequality_op = operatorToAsofInequality(predicate.op);
            if (!asof_inequality_op)
                continue;

            if (asof_predicate_found)
                throw Exception(ErrorCodes::INVALID_JOIN_ON_EXPRESSION, "ASOF join does not support multiple inequality predicates in JOIN ON expression");
            table_join->setAsofInequality(*asof_inequality_op);
            table_join_clauses.front().addKey(predicate.left_node.column_name, predicate.right_node.column_name, /* null_safe_comparison = */ false);
        }
        if (!asof_predicate_found)
            throw Exception(ErrorCodes::INVALID_JOIN_ON_EXPRESSION, "ASOF join requires one inequality predicate in JOIN ON expression");
    }
    else
    {
        for (auto & join_condition : join_info.expression.disjunctive_conditions)
        {
            auto & table_join_clause = table_join_clauses.emplace_back();
            addJoinConditionToTableJoin(join_condition, table_join_clause, &expression_actions.post_join_actions, query_context);
            if (auto left_pre_filter_condition = concatMergeConditions(join_condition.left_filter_conditions, expression_actions.left_pre_join_actions, query_context))
                table_join_clause.analyzer_left_filter_condition_column_name = left_pre_filter_condition.column_name;
            if (auto right_pre_filter_condition = concatMergeConditions(join_condition.right_filter_conditions, expression_actions.right_pre_join_actions, query_context))
                table_join_clause.analyzer_right_filter_condition_column_name = right_pre_filter_condition.column_name;
        }
    }

    JoinActionRef residual_filter_condition(nullptr);
    if (join_info.expression.disjunctive_conditions.empty())
    {
        residual_filter_condition = concatMergeConditions(
            join_info.expression.condition.residual_conditions, expression_actions.post_join_actions, query_context);
    }
    else
    {
        bool need_residual_filter = !join_info.expression.condition.residual_conditions.empty();
        for (const auto & join_condition : join_info.expression.disjunctive_conditions)
        {
            need_residual_filter = need_residual_filter || !join_condition.residual_conditions.empty();
            if (need_residual_filter)
                break;
        }

        if (need_residual_filter)
            residual_filter_condition = buildSingleActionForJoinExpression(join_info.expression, expression_actions, query_context);
    }

    if (residual_filter_condition && canPushDownFromOn(join_info))
    {
        post_filter = residual_filter_condition;
    }
    else if (residual_filter_condition)
    {
        auto & dag = expression_actions.post_join_actions;
        if (is_explain_logical)
            dag = dag.clone();
        auto & outputs = dag.getOutputs();
        for (const auto * node : outputs)
        {
            if (node->result_name == residual_filter_condition.column_name)
            {
                outputs = {node};
                break;
            }
        }

        ExpressionActionsPtr & mixed_join_expression = table_join->getMixedJoinExpression();
        mixed_join_expression = std::make_shared<ExpressionActions>(std::move(dag), ExpressionActionsSettings::fromContext(query_context));
    }

    NameSet required_output_columns_set(required_output_columns.begin(), required_output_columns.end());
    addRequiredInputToOutput(expression_actions.left_pre_join_actions, required_output_columns_set);
    addRequiredInputToOutput(expression_actions.right_pre_join_actions, required_output_columns_set);
    addRequiredInputToOutput(expression_actions.post_join_actions, required_output_columns_set);

    table_join->setInputColumns(
        expression_actions.left_pre_join_actions.getNamesAndTypesList(),
        expression_actions.right_pre_join_actions.getNamesAndTypesList());
    table_join->setUsedColumns(expression_actions.post_join_actions.getRequiredColumnsNames());

    // table_join->setInputColumns(input_headers.at(0).getNamesAndTypesList(), input_headers.at(1).getNamesAndTypesList());
    // table_join->setUsedColumns(output_header->getNames());


    Block right_sample_block(expression_actions.right_pre_join_actions.getResultColumns());
    JoinPtr join_ptr;
    if (join_info.kind == JoinKind::Paste)
        join_ptr = std::make_shared<PasteJoin>(table_join, right_sample_block);
    else
        join_ptr = std::make_shared<HashJoin>(table_join, right_sample_block, settings[Setting::join_any_take_last_row]);

    return join_ptr;
}

}
