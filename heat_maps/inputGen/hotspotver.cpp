/*********************

Hotspot Verify
by Sam Kauffman - Univeristy of Virginia
Verify accuracy of Hotspot input files generated by HotspotEx

*/


#include "64_128.h"
//#include "64_256.h"
//#include "1024_2048.h"
//#include "1024_4096.h"
//#include "1024_8192.h"
//#include "1024_16384.h"

#include <iostream>
#include <fstream>
#include <cstdlib>

#define OUT_SIZE IN_SIZE*MULTIPLIER

using namespace std;

bool verify( char * infName, char * outfName )
{
	const int x = MULTIPLIER;
	double val;
	fstream fs;
	double ** inMatr;

	// allocate 2d array of doubles for input
	inMatr = (double **) malloc( IN_SIZE * sizeof( double * ) );
	for ( int i = 0; i < IN_SIZE; i++ )
		inMatr[i] = (double *) malloc(IN_SIZE * sizeof( double ) );

	// fill input array
	fs.open( infName, ios::in );
	if ( !fs )
		cerr << "Failed to open input file.\n";
	for ( int row = 0; row < IN_SIZE; row++ )
		for ( int col = 0; col < IN_SIZE; col++ )
			fs >> inMatr[row][col];
	fs.close();

	// scan through output file and compare values
	fs.open( outfName, ios::in );
	if ( !fs )
		cerr << "Failed to open output file.\n";
	for ( int row = 0; row < OUT_SIZE; row++ )
		for ( int col = 0; col < OUT_SIZE; col++ )
		{
			fs >> val;
			if ( val != inMatr[row / x][col / x] )
			{
				for ( int i = 0; i < IN_SIZE; i++ )
					free( inMatr[i] );
				free( inMatr );

				return false;
			}
		}
	fs.close();

	for ( int i = 0; i < IN_SIZE; i++ )
		free( inMatr[i] );
	free( inMatr );

	return true;
}

int main( int argc, char * argv[] )
{
	if ( verify( TEMP_IN, TEMP_OUT ) )
		cout << "Temp verified.\n";
	else
		cout << "Temp incorrect.\n";
	if ( verify( POWER_IN, POWER_OUT ) )
		cout << "Power verified.\n";
	else
		cout << "Power incorrect.\n";
}
