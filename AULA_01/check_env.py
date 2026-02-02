import sys
import platform
import importlib.util

def check_lib(lib_name):
    """Verifica se a biblioteca está instalada sem usar pkg_resources"""
    spec = importlib.util.find_spec(lib_name)
    return spec is not None

def run_health_check():
    print("-" * 50)
    print("🚀 INICIANDO CHECK-UP DO AMBIENTE (VERSÃO 2.0)")
    print("-" * 50)

    errors = 0
    warnings = 0

    # 1. Python Version
    py_version = sys.version_info
    if py_version.major == 3 and py_version.minor >= 10:
        print(f"✅ Python {platform.python_version()}: OK")
    else:
        print(f"❌ Python 3.10+ requerido. Você está usando {platform.python_version()}")
        errors += 1

    # 2. Bibliotecas (usando nomes reais de importação)
    libs = {
        "pandas": "pandas",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "sklearn": "scikit-learn"
    }

    print("\n📦 Verificando Bibliotecas:")
    for lib_id, lib_name in libs.items():
        if check_lib(lib_id):
            print(f"  ✅ {lib_name}: Instalado")
        else:
            print(f"  ❌ {lib_name}: NÃO ENCONTRADO!")
            errors += 1

    # 3. venv Check
    if sys.prefix != sys.base_prefix:
        print("\n✅ Ambiente Virtual (venv): Ativo")
    else:
        print("\n⚠️  Ambiente Virtual: NÃO DETECTADO!")
        warnings += 1

    print("\n" + "-" * 50)
    if errors == 0:
        print(f"✨ RESULTADO: {'PRONTO PARA AULA!' if warnings == 0 else 'PRONTO (com avisos)'}")
    else:
        print(f"🆘 RESULTADO: {errors} erro(s). Execute: pip install pandas matplotlib scikit-learn")
    print("-" * 50)

if __name__ == "__main__":
    run_health_check()
