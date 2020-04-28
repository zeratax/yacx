{ pkgs ? import <nixpkgs> {} }:

let fhs = pkgs.buildFHSUserEnv {
        name = "cuda-env";
        targetPkgs = pkgs: with pkgs;
               [ git
                 gitRepo
                 gnupg
                 autoconf
                 curl
		 cmake
                 doxygen
                 procps
                 gnumake
                 gcc7
                 utillinux
                 m4
                 gperf
                 unzip
                 cudatoolkit_10
                 linuxPackages.nvidia_x11
                 libGLU
                 libGL
		 xorg.libXi xorg.libXmu freeglut
                 xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
		 ncurses5
		 stdenv.cc
		 binutils
                 jdk12
                ];
          multiPkgs = pkgs: with pkgs; [ zlib ];
          runScript = "bash";
          profile = ''
                  export CUDA_PATH=${pkgs.cudatoolkit_10}
                  export JAVA_HOME=${pkgs.jdk12.home}
                  export PATH=$CUDA_PATH/bin:$PATH
                  # export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib
		  export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
		  export EXTRA_CCFLAGS="-I/usr/include"
            '';
          };
in pkgs.stdenv.mkDerivation {
   name = "cuda-env-shell";
   nativeBuildInputs = [ fhs ];
   shellHook = "exec cuda-env";
}

