import sys
import platform
import subprocess
from importlib import metadata

# Cores para Terminal
class Colors:
    OK = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def get_lib_version(package_name):
    """
    Tenta obter a versão do pacote instalado via metadata (padrão Python 3.8+).
    Substitui a necessidade de pkg_resources.
    """
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None

def run_health_check():
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD} AI/ML ENVIRONMENT DIAGNOSTIC - PRO VERSION{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")

    errors = 0
    warnings = 0

    # 1. Runtime Check (Python & OS)
    py_version = platform.python_version()
    is_valid_py = sys.version_info.major == 3 and sys.version_info.minor >= 10
    
    status = f"{Colors.OK}OK{Colors.END}" if is_valid_py else f"{Colors.FAIL}REQUER 3.10+{Colors.END}"
    print(f"🔹 Python {py_version} ({platform.system()}): {status}")
    if not is_valid_py: errors += 1

    # 2. Virtual Environment Check
    is_venv = sys.prefix != sys.base_prefix
    status = f"{Colors.OK}Ativo{Colors.END}" if is_venv else f"{Colors.WARNING}NÃO DETECTADO (Global){Colors.END}"
    print(f"🔹 Ambiente Virtual: {status}")
    if not is_venv: warnings += 1

    # 3. Stack de Data Science (Nome do Pacote no PyPI : Nome amigável)
    libs = {
        "numpy": "NumPy",
        "pandas": "Pandas",
        "matplotlib": "Matplotlib",
        "scikit-learn": "Scikit-Learn",
        "scipy": "SciPy"
    }

    print(f"\n{Colors.BOLD} Verificando Dependências Core:{Colors.END}")
    for pkg, name in libs.items():
        version = get_lib_version(pkg)
        if version:
            print(f"  {Colors.OK}✅ {name.ljust(15)} | v{version}{Colors.END}")
        else:
            print(f"  {Colors.FAIL}❌ {name.ljust(15)} | NÃO INSTALADO!{Colors.END}")
            errors += 1

    # 4. Verificação de Performance (NumPy/BLAS)
    try:
        import numpy as np
        # Verifica se o numpy está usando bibliotecas de aceleração
        config = np.show_config
        print(f"\n{Colors.BOLD}⚙️  Otimização de Hardware:{Colors.END}")
        print(f"  ⚡ NumPy SIMD: {np.core.umath.__cpu_features__ if hasattr(np.core, 'umath') else 'N/A'}")
    except:
        pass

    # Relatório Final
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    if errors == 0:
        msg = "AMBIENTE VALIDADO COM SUCESSO!" if warnings == 0 else "AMBIENTE PRONTO (VERIFICAR AVISOS)"
        color = Colors.OK if warnings == 0 else Colors.WARNING
        print(f"{Colors.BOLD}{color}✨ RESULTADO: {msg}{Colors.END}")
    else:
        print(f"{Colors.BOLD}{Colors.FAIL} RESULTADO: {errors} ERRO(S) CRÍTICO(S) ENCONTRADO(S){Colors.END}")
        print(f"Sugestão: pip install {' '.join(libs.keys())}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")

if __name__ == "__main__":
    run_health_check()
