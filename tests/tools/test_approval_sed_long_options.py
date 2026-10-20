"""GNU long-option ownership regressions; command strings are never executed."""
import pytest
from tools import approval, approval_detection


@pytest.fixture(params=[approval.detect_dangerous_command, approval_detection.detect_dangerous_command], ids=['facade', 'definition'])
def detector(request, monkeypatch, tmp_path):
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home' / '.hermes'))
    return request.param


@pytest.mark.parametrize('option', ['--expression', '--expr', '--e', '--file', '--fil', '--fi'])
@pytest.mark.parametrize('attached', [True, False], ids=['attached', 'separate'])
@pytest.mark.parametrize('target', ['~/.bashrc', '~/.hermes/config.yaml', '/etc/hosts'])
def test_abbreviated_expression_keeps_protected_file_operand(detector, option, attached, target):
    value = 's/a/b/' if option.startswith('--e') else 'program.sed'
    argument = option + ('=' if attached else ' ') + value
    positive = 'sed -i ' + argument + ' ' + target
    result = detector(positive)
    assert result[0] is True, (positive, result)
    assert result[1] and result[2]
    for modifier in ['--l=80', '--l 80', '--q', '--sil', '--se', '--follow-s']:
        assert detector('sed ' + modifier + ' ' + argument + ' -i ' + target)[0]
    for command in [
        'sed -i ' + argument + ' ordinary.txt',
        'sed ' + argument + ' ' + target,
        'sed -i ' + option + ('=' if attached else ' ') + target + ' ordinary.txt',
        'sed -i ' + argument + ' ordinary.txt < ' + target,
        'sed -i --l ' + target + ' s/a/b/ ordinary.txt',
        'sed -i ' + argument + ' ordinary.txt # ' + target,
        'printf "%s" \'sed -i ' + argument + ' ' + target + "'",
    ]:
        assert detector(command) == (False, None, None), command
    # Invalid and ambiguous long options must not be guessed from a partial
    # ownership table, including when another option supplied the program.
    for invalid in ['--f=program.sed', '--f program.sed', '--s', '--expressionx=s/a/b/',
                    '--files=program.sed', '--in-places', '--in-placeX', '--EXPR=s/a/b/',
                    '--quiet=1', '--follow-symlinks=' + target]:
        command = 'sed -i -e s/a/b/ ' + invalid + ' ' + target
        assert detector(command) == (False, None, None), (command, detector(command))


@pytest.mark.parametrize('option', ['--in-place', '--in-plac', '--in', '--i'])
@pytest.mark.parametrize('target', ['~/.bashrc', '~/.hermes/config.yaml', '/etc/hosts'])
def test_abbreviated_in_place_keeps_protected_file_operand(detector, option, target):
    for suffix in ['', '=.bak']:
        command = 'sed ' + option + suffix + ' --expr=s/a/b/ ' + target
        result = detector(command)
        assert result[0] is True, (command, result)
        assert result[1] and result[2]
        assert detector('sed ' + option + suffix + ' --expr=s/a/b/ ordinary.txt') == (False, None, None)
    # Optional backup suffixes are attached only, not the next operand.
    assert detector('sed ' + option + ' --expr=s/a/b/ ' + target + ' ordinary.txt')[0]
    assert detector('sed -- ' + option + ' s/a/b/ ' + target) == (False, None, None)
    assert detector('sed ' + option + ' --expr=s/a/b/ -- ' + target)[0]
