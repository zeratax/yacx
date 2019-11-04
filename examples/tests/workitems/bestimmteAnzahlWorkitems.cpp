
// Copyright 2019 André Hodapp
#include <stdio.h>
#include <stdlib.h>

void bestimmeAnzahlWorkitems(
    int *workitems, int *bloecke, const int &registerProSIMD,
    const int &sharedSpeicherplatzProSIMD,
    const int &Max_Anz_Workitems_Block,
    const int &registerProWorkitem,
    const int &sharedSpeicherplatzProWorkitem,
    const int &Max_Anz_Blocke, const int &workitemsInsgesamt = 1024) {
  // setze die Anzahl der Workitems in eienm Block
  int itemsProBlock = Max_Anz_Workitems_Block;
  // Bestimme, ob Register passen
  if (registerProSIMD / (registerProWorkitem * itemsProBlock) < 1) {
    itemsProBlock = registerProSIMD / registerProWorkitem;
  }
  // Bestimme, ob shared Speicher passt
  if (sharedSpeicherplatzProSIMD /
          (sharedSpeicherplatzProWorkitem * itemsProBlock) <
      1) {
    itemsProBlock = sharedSpeicherplatzProSIMD / sharedSpeicherplatzProWorkitem;
  }
  // Bestimme, ob es ein Vielfaches von 32 ist
  if (itemsProBlock % 32 != 0) {
    itemsProBlock = itemsProBlock - (itemsProBlock % 32);
  }

  // setze die Anzahl der Bloecke
  int anzahl_bloecke = (workitemsInsgesamt + itemsProBlock - 1) / itemsProBlock;
  // Ueberpruefe, ob es unter der maximalen Anzahl Bloecke liegt
  if (anzahl_bloecke > Max_Anz_Blocke) {
    anzahl_bloecke = Max_Anz_Blocke;
    printf("Max_Anz_Bloecke passt nicht\n");
    printf("Error: Die angeforderten Anzahl an Workitems konnten nicht "
           "gestartet werden, sondern nur %i viele\n",
           anzahl_bloecke);
    printf("Somit konnten insgesamt nur %i viele Workitems gestartet werden\n",
           anzahl_bloecke * itemsProBlock);
  }
  // Ueberpruefe, ob Register passen
  if (anzahl_bloecke >
      registerProSIMD / (registerProWorkitem * itemsProBlock)) {
    anzahl_bloecke = registerProSIMD / (registerProWorkitem * itemsProBlock);
    printf("Registeranzahl passt nicht\n");
    printf("Error: Die angeforderten Anzahl an Workitems konnten nicht "
           "gestartet werden, sondern nur %i viele\n",
           anzahl_bloecke);
    printf("Somit konnten insgesamt nur %i viele Workitems gestartet werden\n",
           anzahl_bloecke * itemsProBlock);
  }
  // Ueberpruefe, ob shared Speicher passt
  if (anzahl_bloecke > sharedSpeicherplatzProSIMD /
                           (sharedSpeicherplatzProWorkitem * itemsProBlock)) {
    anzahl_bloecke = sharedSpeicherplatzProSIMD /
                     (sharedSpeicherplatzProWorkitem * itemsProBlock);
    printf("shared Speicher passt nicht\n");
    printf("Error: Die angeforderten Anzahl an Workitems konnten nicht "
           "gestartet werden, sondern nur %i viele\n",
           anzahl_bloecke);
    printf("Somit konnten insgesamt nur %i viele Workitems gestartet werden\n",
           anzahl_bloecke * itemsProBlock);
  }

  // setze die Werte in die Referenzen
  *workitems = itemsProBlock;
  *bloecke = anzahl_bloecke;
}

void test() {
  int workitems = 0;
  int bloecke = 0;

  bestimmeAnzahlWorkitems(&workitems, &bloecke, 16384, 16384, 1024, 14, 18, 8);
  printf("workitems: %i bloecke: %i\n", workitems, bloecke);

  bestimmeAnzahlWorkitems(&workitems, &bloecke, 16384, 16384, 512, 16, 16, 8);
  printf("workitems: %i bloecke: %i\n", workitems, bloecke);
  bestimmeAnzahlWorkitems(&workitems, &bloecke, 16384, 16384, 1024, 16, 16, 8);
  printf("workitems: %i bloecke: %i\n", workitems, bloecke);
  bestimmeAnzahlWorkitems(&workitems, &bloecke, 16384, 16384, 512, 16, 16, 8,
                          1000000);
  printf("workitems: %i bloecke: %i\n", workitems, bloecke);
}

int main(void) { test(); }
