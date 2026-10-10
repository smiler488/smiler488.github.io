# smiler488 Lab MCP server

A local [Model Context Protocol](https://modelcontextprotocol.io) server that
lets AI assistants (Claude Code, Claude Desktop and other MCP clients) call the
science functions behind the App Lab at <https://smiler488.github.io/app>.

It imports the same code the website runs (`src/lib/science`), so an assistant
gets exactly the numbers the web tools show. No dependencies, no network
access, nothing leaves your machine.

## Tools

| Tool             | What it computes                                                                                               | Same as                           |
| ---------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| `field_area`     | Area of a lat/lng field boundary in m², ha and mu, plus its centroid (WGS 84, validated against GeographicLib) | Land Surveyor                     |
| `solar_position` | Sun elevation and azimuth for a place and moment (NOAA algorithm, validated against NREL SPA)                  | Sensor Recorder                   |
| `leaf_brdf`      | Leaf BRDF in the principal plane, Deng et al. 2025, within the study's fitting bounds                          | BRDF explorer on the project page |

Inputs are validated. Out-of-range values come back as tool errors the
assistant can read and correct, not as crashes.

## Use it

Requires Node.js 20 or later and a clone of this repository.

Claude Code:

```bash
claude mcp add smiler488-lab -- node /path/to/smiler488.github.io/packages/lab-mcp/server.mjs
```

Claude Desktop or another MCP client, in its server configuration:

```json
{
  "mcpServers": {
    "smiler488-lab": {
      "command": "node",
      "args": ["/path/to/smiler488.github.io/packages/lab-mcp/server.mjs"]
    }
  }
}
```

Then ask, for example: "Use smiler488-lab to compute the area of a field with
these vertices…" or "What is the sun elevation in Beijing at noon on the June
solstice (UTC+8)?"

## Tests

From the repository root:

```bash
npm run test:science
```

This runs the science-layer validation tests (against NREL SPA, GeographicLib
and the BRDF study's fitting code) and a protocol test that drives the server over stdio the way an MCP
client does.

## Citing

The BRDF model: Deng, L., Yu, L. X., Mao, L., Wang, Y., Guo, X., Wang, M.,
Zhang, Y., Song, Q., & Zhu, X.-G. (2025). Leaf bidirectional reflectance
distribution function (BRDF) prediction with phenotypic traits in four species.
_Plant Phenomics, 7_(4), 100135. <https://doi.org/10.1016/j.plaphe.2025.100135>
