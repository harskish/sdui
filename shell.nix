let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell.override { stdenv = pkgs.stdenvNoCC; } {
  buildInputs = with pkgs.python3.pkgs; [
    pkgs.rustup
    pkgs.libiconv
    pkgs.pkg-config
    pkgs.darwin.apple_sdk.frameworks.Security
  ];
  shellHook = ''
    echo ""
    echo "Current conda env:"
    echo "$(conda env list | grep '*')"
    echo "Run:"
    echo "conda env update -f env_mac.yml --prune"
  '';
}
