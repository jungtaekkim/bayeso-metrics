STR_VERSION = '0.1.1'


def test_version_bayeso_metrics():
    import bayeso_metrics
    assert bayeso_metrics.__version__ == STR_VERSION

def test_version_setup():
    try:
        import importlib
        assert importlib.metadata.version('bayeso_metrics') == STR_VERSION
    except:
        import pkg_resources
        assert pkg_resources.require('bayeso_metrics')[0].version == STR_VERSION
