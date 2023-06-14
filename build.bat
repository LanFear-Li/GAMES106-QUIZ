cd build || goto :error

cmake -G "Visual Studio 16 2019" -A "x64" .. || goto :error
cd ..