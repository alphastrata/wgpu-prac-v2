{
  description = "CUDA dev shell environment with Rust support";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, rust-overlay }:
    let
      overlays = [
        (import rust-overlay)
        (self: super: {
          rustToolchain = super.rust-bin.nightly.latest.default;
        })
      ];

      allSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      forAllSystems = f: nixpkgs.lib.genAttrs allSystems (system: f {
        pkgs = import nixpkgs {
          inherit overlays system;
          config.allowUnfree = true;
        };
      });
    in
    {
      devShells = forAllSystems ({ pkgs }: {
        default = pkgs.mkShell {
          packages = (with pkgs; [
            alsa-lib
            cargo-nextest
            lld
            openssl
            pkg-config
            rustToolchain
            udev
            vulkan-headers
            vulkan-loader
            vulkan-tools
            vulkan-validation-layers
            xorg.libX11
            xorg.libXcursor
            xorg.libXi
            xorg.libXrandr
            pkgs.gcc11
            cudatoolkit
          ]) ++ pkgs.lib.optionals pkgs.stdenv.isDarwin (with pkgs; [
            libiconv
          ]);

          shellHook = ''
            export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.lib.makeLibraryPath [
              pkgs.udev
              pkgs.vulkan-loader
              pkgs.openssl
              pkgs.cudatoolkit
            ]}"

            export CUDA_PATH=${pkgs.cudatoolkit}
            export LIBRARY_PATH=$CUDA_PATH/lib:$LIBRARY_PATH
            export LD_LIBRARY_PATH=$CUDA_PATH/lib:$LD_LIBRARY_PATH
           
            export PATH=${pkgs.gcc11}/bin:$PATH  # Prioritize GCC 11 in the PATH
            export CC=${pkgs.gcc11}/bin/gcc
            export CXX=${pkgs.gcc11}/bin/g++
            rustup default nightly
          '';
        };
      });
    };
}
