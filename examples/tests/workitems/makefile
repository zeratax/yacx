# von André Hodapp

C++FLAGS := -Wall -Wextra -pedantic -Werror -std=c++14 -fPIC 
out: bestimmeAnzahlWorkitems.o 
	g++ $(C++FLAGS) -o bestimmeAnzahlWorkitems bestimmeAnzahlWorkitems.o 

bestimmeAnzahlWorkitems.o: bestimmeAnzahlWorkitems.cpp
	g++ $(C++FLAGS) -c bestimmeAnzahlWorkitems.cpp 

clean:
	rm *.o out 
run:
	./out
