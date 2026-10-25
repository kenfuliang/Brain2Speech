from reproduction.native_acceleration_probe import interpreter_path


def test_virtual_environment_interpreter_symlink_is_not_resolved(tmp_path):
    base = tmp_path / 'base-python'
    base.touch()
    venv = tmp_path / 'venv-python'
    venv.symlink_to(base)
    assert interpreter_path(venv) == str(venv)
    assert interpreter_path(venv) != str(venv.resolve())
