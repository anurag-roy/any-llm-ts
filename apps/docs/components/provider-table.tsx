import providerData from "@/lib/provider-data.json";

const capabilityColumns = [
  ["completion", "Chat"],
  ["streaming", "Stream"],
  ["embedding", "Embed"],
  ["responses", "Responses"],
  ["vision", "Vision"],
  ["reasoning", "Reasoning"],
] as const;

export function ProviderTable() {
  return (
    <div className="my-6 overflow-x-auto rounded-lg border">
      <table className="w-full min-w-[900px] text-sm">
        <thead className="bg-fd-muted/50 text-left">
          <tr>
            <th className="px-3 py-2 font-medium">Provider</th>
            <th className="px-3 py-2 font-medium">Environment variable</th>
            {capabilityColumns.map(([, label]) => (
              <th key={label} className="px-3 py-2 text-center font-medium">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {providerData.map((provider) => (
            <tr key={provider.name} className="border-t">
              <td className="px-3 py-2">
                <a href={provider.documentationUrl} target="_blank" rel="noreferrer">
                  <code>{provider.name}</code>
                </a>
              </td>
              <td className="px-3 py-2 text-fd-muted-foreground">
                {provider.envApiKey ? <code>{provider.envApiKey}</code> : "Not required"}
              </td>
              {capabilityColumns.map(([capability]) => (
                <td key={capability} className="px-3 py-2 text-center">
                  <span className={provider.capabilities[capability] ? "text-emerald-600" : "text-fd-muted-foreground"}>
                    {provider.capabilities[capability] ? "✓" : "—"}
                  </span>
                  <span className="sr-only">
                    {provider.capabilities[capability] ? "Supported" : "Not supported"}
                  </span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
