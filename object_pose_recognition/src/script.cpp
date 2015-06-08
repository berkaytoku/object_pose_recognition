#include <iostream>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include <cstdlib>

using namespace std;

string class1 = "../data/OPR-Dataset/class1/data/";
string class2 = "../data/OPR-Dataset/class2/data/";
string class3 = "../data/OPR-Dataset/class3/data/";
string class4 = "../data/OPR-Dataset/class4/data/";
string class5 = "../data/OPR-Dataset/class5/data/";
string class6 = "../data/OPR-Dataset/class6/data/";
string class7 = "../data/OPR-Dataset/class7/data/";
string class8 = "../data/OPR-Dataset/class8/data/";
string class9 = "../data/OPR-Dataset/class9/data/";
string class10 = "../data/OPR-Dataset/class10/data/";
string class11 = "../data/OPR-Dataset/class11/data/";
string class12 = "../data/OPR-Dataset/class12/data/";
string class13 = "../data/OPR-Dataset/class13/data/";
string class14 = "../data/OPR-Dataset/class14/data/";
string class15 = "../data/OPR-Dataset/class15/data/"; 

int main() {
	//srand ( time(NULL) );
	string firstOutput, secondOutput, thirdOutput, output;
	int tag = 0;
	ofstream outputFile;
	outputFile.open("../data/trainingData.txt");
	//int i = 1;
	for(int i = 0; i<15000; i++){
		if(i>=0 && i<1000){  // class 1
			firstOutput = class1 + "color" + static_cast<ostringstream*>( &(ostringstream() << i) )->str() + ".jpg";
			secondOutput = class1 + "color" + static_cast<ostringstream*>( &(ostringstream() << i+1) )->str() + ".jpg";
			int classRand;
			if(i%2 == 1){
				classRand = (rand()%15) + 1;
				while(classRand==1){
					classRand = (rand()%15) + 1;
				}
				int sampleRand = rand()%1000;
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			else{
				classRand = 1;
				int sampleRand = rand()%1000;
				while(sampleRand==i || sampleRand == i+1){
					sampleRand = rand()%1000;
				}
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			tag = 1;
			
			output = firstOutput + " " + secondOutput + " " + thirdOutput + " " + static_cast<ostringstream*>( &(ostringstream() << tag) )->str() + "\n";
			outputFile<<output;
		}
		
		if(i>=1000 && i<2000){ //class2
			firstOutput = class2 + "color" + static_cast<ostringstream*>( &(ostringstream() << i-1000) )->str() + ".jpg";
			secondOutput = class2 + "color" + static_cast<ostringstream*>( &(ostringstream() << i+1-1000) )->str() + ".jpg";
			int classRand;
			if(i%2 == 1){
				classRand = (rand()%15) + 1;
				while(classRand==2){
					classRand = (rand()%15) + 1;
				}
				int sampleRand = rand()%1000;
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			else{
				classRand = 2;
				int sampleRand = rand()%1000;
				while(sampleRand==i-1000 || sampleRand == i+1-1000){
					sampleRand = rand()%1000;
				}
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			tag = 2;
			
			output = firstOutput + " " + secondOutput + " " + thirdOutput + " " + static_cast<ostringstream*>( &(ostringstream() << tag) )->str() + "\n";
			outputFile<<output;
		}
		
		if(i>=2000 && i<3000){ //class3
			firstOutput = class3 + "color" + static_cast<ostringstream*>( &(ostringstream() << i-2000) )->str() + ".jpg";
			secondOutput = class3 + "color" + static_cast<ostringstream*>( &(ostringstream() << i+1-2000) )->str() + ".jpg";
			int classRand;
			if(i%2 == 1){
				classRand = (rand()%15) + 1;
				while(classRand==3){
					classRand = (rand()%15) + 1;
				}
				int sampleRand = rand()%1000;
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			else{
				classRand = 3;
				int sampleRand = rand()%1000;
				while(sampleRand==i-2000 || sampleRand == i+1-2000){
					sampleRand = rand()%1000;
				}
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			tag = 3;
			
			output = firstOutput + " " + secondOutput + " " + thirdOutput + " " + static_cast<ostringstream*>( &(ostringstream() << tag) )->str() + "\n";
			outputFile<<output;
		}
		
		if(i>=3000 && i<4000){  //class4
			firstOutput = class4 + "color" + static_cast<ostringstream*>( &(ostringstream() << i-3000) )->str() + ".jpg";
			secondOutput = class4 + "color" + static_cast<ostringstream*>( &(ostringstream() << i+1-3000) )->str() + ".jpg";
			int classRand;
			if(i%2 == 1){
				classRand = (rand()%15) + 1;
				while(classRand==4){
					classRand = (rand()%15) + 1;
				}
				int sampleRand = rand()%1000;
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			else{
				classRand = 4;
				int sampleRand = rand()%1000;
				while(sampleRand==i-3000 || sampleRand == i+1-3000){
					sampleRand = rand()%1000;
				}
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			tag = 4;
			
			output = firstOutput + " " + secondOutput + " " + thirdOutput + " " + static_cast<ostringstream*>( &(ostringstream() << tag) )->str() + "\n";
			outputFile<<output;
		}
		
		if(i>=4000 && i<5000){ //class5
			firstOutput = class5 + "color" + static_cast<ostringstream*>( &(ostringstream() << i-4000) )->str() + ".jpg";
			secondOutput = class5 + "color" + static_cast<ostringstream*>( &(ostringstream() << i+1-4000) )->str() + ".jpg";
			int classRand;
			if(i%2 == 1){
				classRand = (rand()%15) + 1;
				while(classRand==5){
					classRand = (rand()%15) + 1;
				}
				int sampleRand = rand()%1000;
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			else{
				classRand = 5;
				int sampleRand = rand()%1000;
				while(sampleRand==i-4000 || sampleRand == i+1-4000){
					sampleRand = rand()%1000;
				}
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			tag = 5;
			
			output = firstOutput + " " + secondOutput + " " + thirdOutput + " " + static_cast<ostringstream*>( &(ostringstream() << tag) )->str() + "\n";
			outputFile<<output;
		}
		
		if(i>=5000 && i<6000){ //class6
			firstOutput = class6 + "color" + static_cast<ostringstream*>( &(ostringstream() << i-5000) )->str() + ".jpg";
			secondOutput = class6 + "color" + static_cast<ostringstream*>( &(ostringstream() << i+1-5000) )->str() + ".jpg";
			int classRand;
			if(i%2 == 1){
				classRand = (rand()%15) + 1;
				while(classRand==6){
					classRand = (rand()%15) + 1;
				}
				int sampleRand = rand()%1000;
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			else{
				classRand = 6;
				int sampleRand = rand()%1000;
				while(sampleRand==i-5000 || sampleRand == i+1-5000){
					sampleRand = rand()%1000;
				}
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			tag = 6;
			
			output = firstOutput + " " + secondOutput + " " + thirdOutput + " " + static_cast<ostringstream*>( &(ostringstream() << tag) )->str() + "\n";
			outputFile<<output;
		}
		
		if(i>=6000 && i<7000){ //class7
			firstOutput = class7 + "color" + static_cast<ostringstream*>( &(ostringstream() << i-6000) )->str() + ".jpg";
			secondOutput = class7 + "color" + static_cast<ostringstream*>( &(ostringstream() << i+1-6000) )->str() + ".jpg";
			int classRand;
			if(i%2 == 1){
				classRand = (rand()%15) + 1;
				while(classRand==7){
					classRand = (rand()%15) + 1;
				}
				int sampleRand = rand()%1000;
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			else{
				classRand = 7;
				int sampleRand = rand()%1000;
				while(sampleRand==i-6000 || sampleRand == i+1-6000){
					sampleRand = rand()%1000;
				}
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			tag = 7;
			
			output = firstOutput + " " + secondOutput + " " + thirdOutput + " " + static_cast<ostringstream*>( &(ostringstream() << tag) )->str() + "\n";
			outputFile<<output;
		}
		
		if(i>=7000 && i<8000){ //class8
			firstOutput = class8 + "color" + static_cast<ostringstream*>( &(ostringstream() << i-7000) )->str() + ".jpg";
			secondOutput = class8 + "color" + static_cast<ostringstream*>( &(ostringstream() << i+1-7000) )->str() + ".jpg";
			int classRand;
			if(i%2 == 1){
				classRand = (rand()%15) + 1;
				while(classRand==8){
					classRand = (rand()%15) + 1;
				}
				int sampleRand = rand()%1000;
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			else{
				classRand = 8;
				int sampleRand = rand()%1000;
				while(sampleRand==i-7000 || sampleRand == i+1-7000){
					sampleRand = rand()%1000;
				}
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			tag = 8;
			
			output = firstOutput + " " + secondOutput + " " + thirdOutput + " " + static_cast<ostringstream*>( &(ostringstream() << tag) )->str() + "\n";
			outputFile<<output;
		}
		
		if(i>=8000 && i<9000){ //class9
			firstOutput = class9 + "color" + static_cast<ostringstream*>( &(ostringstream() << i-8000) )->str() + ".jpg";
			secondOutput = class9 + "color" + static_cast<ostringstream*>( &(ostringstream() << i+1-8000) )->str() + ".jpg";
			int classRand;
			if(i%2 == 1){
				classRand = (rand()%15) + 1;
				while(classRand==9){
					classRand = (rand()%15) + 1;
				}
				int sampleRand = rand()%1000;
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			else{
				classRand = 9;
				int sampleRand = rand()%1000;
				while(sampleRand==i-8000 || sampleRand == i+1-8000){
					sampleRand = rand()%1000;
				}
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			tag = 9;
			
			output = firstOutput + " " + secondOutput + " " + thirdOutput + " " + static_cast<ostringstream*>( &(ostringstream() << tag) )->str() + "\n";
			outputFile<<output;
		}
		
		if(i>=9000 && i<10000){ //class10
			firstOutput = class10 + "color" + static_cast<ostringstream*>( &(ostringstream() << i-9000) )->str() + ".jpg";
			secondOutput = class10 + "color" + static_cast<ostringstream*>( &(ostringstream() << i+1-9000) )->str() + ".jpg";
			int classRand;
			if(i%2 == 1){
				classRand = (rand()%15) + 1;
				while(classRand==10){
					classRand = (rand()%15) + 1;
				}
				int sampleRand = rand()%1000;
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			else{
				classRand = 10;
				int sampleRand = rand()%1000;
				while(sampleRand==i-9000 || sampleRand == i+1-9000){
					sampleRand = rand()%1000;
				}
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			tag = 10;
			
			output = firstOutput + " " + secondOutput + " " + thirdOutput + " " + static_cast<ostringstream*>( &(ostringstream() << tag) )->str() + "\n";
			outputFile<<output;
		}
		
		if(i>=10000 && i<11000){ //class11
			firstOutput = class11 + "color" + static_cast<ostringstream*>( &(ostringstream() << i-10000) )->str() + ".jpg";
			secondOutput = class11 + "color" + static_cast<ostringstream*>( &(ostringstream() << i+1-10000) )->str() + ".jpg";
			int classRand;
			if(i%2 == 1){
				classRand = (rand()%15) + 1;
				while(classRand==11){
					classRand = (rand()%15) + 1;
				}
				int sampleRand = rand()%1000;
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			else{
				classRand = 11;
				int sampleRand = rand()%1000;
				while(sampleRand==i-10000 || sampleRand == i+1-10000){
					sampleRand = rand()%1000;
				}
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			tag = 11;
			
			output = firstOutput + " " + secondOutput + " " + thirdOutput + " " + static_cast<ostringstream*>( &(ostringstream() << tag) )->str() + "\n";
			outputFile<<output;
		}
		
		if(i>=11000 && i<12000){  //class12
			firstOutput = class12 + "color" + static_cast<ostringstream*>( &(ostringstream() << i-11000) )->str() + ".jpg";
			secondOutput = class12 + "color" + static_cast<ostringstream*>( &(ostringstream() << i+1-11000) )->str() + ".jpg";
			int classRand;
			if(i%2 == 1){
				classRand = (rand()%15) + 1;
				while(classRand==12){
					classRand = (rand()%15) + 1;
				}
				int sampleRand = rand()%1000;
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			else{
				classRand = 12;
				int sampleRand = rand()%1000;
				while(sampleRand==i-11000 || sampleRand == i+1-11000){
					sampleRand = rand()%1000;
				}
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			tag = 12;
			
			output = firstOutput + " " + secondOutput + " " + thirdOutput + " " + static_cast<ostringstream*>( &(ostringstream() << tag) )->str() + "\n";
			outputFile<<output;
		}
		
		if(i>=12000 && i<13000){  //class13
			firstOutput = class13 + "color" + static_cast<ostringstream*>( &(ostringstream() << i-12000) )->str() + ".jpg";
			secondOutput = class13 + "color" + static_cast<ostringstream*>( &(ostringstream() << i+1-12000) )->str() + ".jpg";
			int classRand;
			if(i%2 == 1){
				classRand = (rand()%15) + 1;
				while(classRand==13){
					classRand = (rand()%15) + 1;
				}
				int sampleRand = rand()%1000;
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			else{
				classRand = 13;
				int sampleRand = rand()%1000;
				while(sampleRand==i-12000 || sampleRand == i+1-12000){
					sampleRand = rand()%1000;
				}
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			tag = 13;
			
			output = firstOutput + " " + secondOutput + " " + thirdOutput + " " + static_cast<ostringstream*>( &(ostringstream() << tag) )->str() + "\n";
			outputFile<<output;
		}
		
		if(i>=13000 && i<14000){ //class14
			firstOutput = class14 + "color" + static_cast<ostringstream*>( &(ostringstream() << i-13000) )->str() + ".jpg";
			secondOutput = class14 + "color" + static_cast<ostringstream*>( &(ostringstream() << i+1-13000) )->str() + ".jpg";
			int classRand;
			if(i%2 == 1){
				classRand = (rand()%15) + 1;
				while(classRand==14){
					classRand = (rand()%15) + 1;
				}
				int sampleRand = rand()%1000;
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			else{
				classRand = 14;
				int sampleRand = rand()%1000;
				while(sampleRand==i-13000 || sampleRand == i+1-13000){
					sampleRand = rand()%1000;
				}
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			tag = 14;
			
			output = firstOutput + " " + secondOutput + " " + thirdOutput + " " + static_cast<ostringstream*>( &(ostringstream() << tag) )->str() + "\n";
			outputFile<<output;
		}
		
		if(i>=14000 && i<15000){ //class15
			firstOutput = class15 + "color" + static_cast<ostringstream*>( &(ostringstream() << i-14000) )->str() + ".jpg";
			secondOutput = class15 + "color" + static_cast<ostringstream*>( &(ostringstream() << i+1-14000) )->str() + ".jpg";
			int classRand;
			if(i%2 == 1){
				classRand = (rand()%15) + 1;
				while(classRand==15){
					classRand = (rand()%15) + 1;
				}
				int sampleRand = rand()%1000;
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			else{
				classRand = 15;
				int sampleRand = rand()%1000;
				while(sampleRand==i-14000 || sampleRand == i+1-14000){
					sampleRand = rand()%1000;
				}
				thirdOutput = "../data/OPR-Dataset/class" + static_cast<ostringstream*>( &(ostringstream() << classRand) )->str() + "/data/" + "color" + static_cast<ostringstream*>( &(ostringstream() << sampleRand) )->str() + ".jpg";
			}
			tag = 15;
			
			output = firstOutput + " " + secondOutput + " " + thirdOutput + " " + static_cast<ostringstream*>( &(ostringstream() << tag) )->str() + "\n";
			outputFile<<output;
		}
	}
	
	outputFile.close();
	return 0;	
}
