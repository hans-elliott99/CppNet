all:
	g++ -fdiagnostics-color=always -g main.cpp -o main.exe -O3 data/*.cpp net/*.cpp
