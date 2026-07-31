def is_jupyter_env():
    """Check if code is running in a Jupyter notebook environment."""
    try:
        # Check for IPython kernel
        from IPython import get_ipython
        if get_ipython() is None:
            return False

        # Check if it's specifically a notebook (not just IPython terminal)
        if 'IPKernelApp' in get_ipython().config:
            return True
    except (ImportError, AttributeError):
        pass

    return False


_ASYNCIO_LOOP_PATCHED = False


def ensure_asyncio_loop():
    """Ensure that the asyncio event loop is properly set up for Jupyter notebooks."""
    global _ASYNCIO_LOOP_PATCHED
    if not _ASYNCIO_LOOP_PATCHED and is_jupyter_env():
        import nest_asyncio
        nest_asyncio.apply()
        _ASYNCIO_LOOP_PATCHED = True
