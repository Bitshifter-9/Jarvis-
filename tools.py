import os
import datetime
import subprocess
import webbrowser


def open_app(app_name):
    apps = {
        "chrome": "open -a 'Google Chrome'",
        "safari": "open -a Safari",
        "vscode": "open -a 'Visual Studio Code'",
        "terminal": "open -a Terminal"
    }

    cmd = apps.get(app_name.lower())
    if cmd:
        try:
            result = os.system(cmd)
            if result == 0:
                return f"Opened {app_name}"
            else:
                return f"Failed to open {app_name}"
        except Exception as e:
            return f"Error opening {app_name}: {str(e)}"
    return "App not found"
def get_time():
    return datetime.datetime.now().strftime("Current time is %I:%M %p")

def search_google(query):
    url = f"https://www.google.com/search?q={query}"
    webbrowser.open(url)
    return f"Searching Google for {query}"

def run_command(cmd):
    try:
        output = subprocess.check_output(cmd, shell=True, text=True)
        return output[:500]
    except Exception as e:
        return str(e)