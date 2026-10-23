{
  description = "Python Dev";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      projectRoot = ./.;
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      pkgsFor = system: (nixpkgs.legacyPackages.${system});
      pythonFor = system: (pkgsFor system).python314.withPackages (ps: [ ps.matplotlib ps.pytest ]);
    in
    {
      devShells = nixpkgs.lib.genAttrs systems (
        system:
        let
          pkgs = pkgsFor system;
          python = pythonFor system;
        in
        {
          default = pkgs.mkShell {
            packages = [
              python
              pkgs.ruff
            ];
            shellHook = ''
              export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
            '';
          };
        }
      );

      # terminal app for running tests
      # nix run ".#test"
      apps = nixpkgs.lib.genAttrs systems (
        system:
        let
          pkgs = pkgsFor system;
          python = pythonFor system;

          testRunner = pkgs.writeShellApplication {
            name = "test-runner";
            runtimeInputs = [ python ];
            text = ''
              export PYTHONPATH="${projectRoot}/src:''${PYTHONPATH:-}"
              # pytest without cacheprovider as nix store is readonly
              pytest -p no:cacheprovider "${projectRoot}/tests"
            '';
          };
        in
        {
          test = {
            type = "app";
            program = "${testRunner}/bin/test-runner";
          };
        }
      );
    };
}
