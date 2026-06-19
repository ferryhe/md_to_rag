from __future__ import annotations

from .schemas import COMMAND_INPUT_MODELS, COMMAND_OUTPUT_MODELS, CommandName, ToolMetadata


_TOOL_DESCRIPTIONS = {
    CommandName.INIT: "Prepare project artifact layout metadata.",
    CommandName.INGEST: "Read Markdown sources into document artifact metadata.",
    CommandName.CHUNK: "Create chunk artifact metadata from document manifests.",
    CommandName.EMBED: "Create embedding artifact metadata from chunk artifacts.",
    CommandName.INDEX: "Create index artifact metadata from embedding artifacts.",
    CommandName.QUERY: "Return deterministic local retrieval results from index artifacts.",
    CommandName.INSPECT: "Inspect artifact status metadata.",
    CommandName.DIFF: "Report artifact-chain drift without mutating artifacts.",
    CommandName.REBUILD: "Run ingest, chunk, embed, and index with default artifact paths.",
}


def list_tools() -> list[ToolMetadata]:
    return [
        ToolMetadata(
            name=f"md_to_rag_{command.value}",
            command=command,
            description=_TOOL_DESCRIPTIONS[command],
            input_schema=COMMAND_INPUT_MODELS[command].model_json_schema(),
            output_schema=COMMAND_OUTPUT_MODELS[command].model_json_schema(),
        )
        for command in CommandName
    ]
