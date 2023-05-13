#pragma once

#include <string>
#include <Core/Protocol.h>
#include <IO/ConnectionTimeouts.h>

namespace Poco::Util
{
class AbstractConfiguration;
}

namespace DB
{
struct ConnectionParameters
{
    std::string host;
    UInt16 port{};
    std::string default_database;
    std::string user;
    std::string password;
    std::string quota_key;
    Protocol::Secure security = Protocol::Secure::Disable;
    Protocol::Compression compression = Protocol::Compression::Enable;
    ConnectionTimeouts timeouts;

    ConnectionParameters() = default;
    // We don't take database from config, as it can be changed after query execution
    ConnectionParameters(const Poco::Util::AbstractConfiguration & config, const std::string & database);
    ConnectionParameters(
        const Poco::Util::AbstractConfiguration & config, const std::string & database, std::string host, std::optional<UInt16> port
    );

    static UInt16 getPortFromConfig(const Poco::Util::AbstractConfiguration & config);

    /// Ask to enter the user's password if password option contains this value.
    /// "\n" is used because there is hardly a chance that a user would use '\n' as password.
    static constexpr std::string_view ASK_PASSWORD = "\n";

    static ConnectionParameters createForEmbedded(const String & user, const String & database);
};

}
