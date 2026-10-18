{
  description = "Python environment with matplotlib";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python3.withPackages (ps: with ps; [
          matplotlib
          numpy
          ipykernel
          jupyter
        ]);
      in {
        devShells.default = pkgs.mkShell {
          packages = [ python ];
        };

        packages.default = python;
      });
}
