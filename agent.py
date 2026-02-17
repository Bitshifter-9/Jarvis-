from tools import open_app, get_time, search_google, run_command

TOOLS = {
    "open_app": open_app,
    "get_time": lambda _: get_time(),
    "search_google": search_google,
    "run_command": run_command
}

def execute_tool(tool_name, arg):
    tool = TOOLS.get(tool_name)
    if not tool:
        return "Tool not found"

    try:
        return tool(arg)
    except Exception as e:
        return str(e)
