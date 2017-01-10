#!/bin/bash

if [[ -a .git/shallow ]]; then git fetch --unshallow; fi
# julia -e 'Pkg.clone(pwd()); Pkg.build("MXNet"); Pkg.test("MXNet"; coverage=true)'
julia -e 'Pkg.clone(pwd()); Pkg.build("MXNet")'
/home/travis/julia/bin/julia -Cx86-64 -J/home/travis/julia/lib/julia/sys.so --compile=yes --depwarn=yes --check-bounds=yes --code-coverage=user --inline=no --color=no --compilecache=yes /home/travis/.julia/v0.5/MXNet/test/runtests.jl
