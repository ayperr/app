"""
Compatibility shim: this deployment's saved Streamlit Cloud entry point is
still `streamlit_app.py` from before the file was renamed to `app.py` (the
name the assignment requires). Rather than duplicate the app's logic here,
this just runs the real entry point.
"""
import runpy

runpy.run_path("app.py", run_name="__main__")
