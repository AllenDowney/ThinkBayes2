{ pkgs ? import <nixpkgs> {} }:

let
  empiricalDist = pkgs.python36Packages.buildPythonPackage rec {
    pname = "empiricaldist";
    version = "0.3.9";

    src = pkgs.python36Packages.fetchPypi {
      inherit pname version;
      sha256 = "8b3a09afa0a20788050bad6c78dc4198e23d1a6970b5576d11be240fdbdf9d80";
    };

    propagatedBuildInputs = with pkgs.python36Packages; [
      jupyter
      matplotlib
      notebook
      numpy
      pandas
    ];

    doCheck = false;

  };

  customPython = pkgs.python36.buildEnv.override {
    extraLibs = [ empiricalDist ];
  };
in

pkgs.mkShell {
  buildInputs = [ empiricalDist ];
}
