#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <cstring>
#include <algorithm>
#include <vector>
#include <ctime>
#include <fstream>

using namespace std;

vector<string> dataVector;

bool compare_by_word(const string lhs, const string rhs) {
    return lhs < rhs;
}

int setDataVector(string folder) {
		
		char flag = folder[21];
		cout<<flag<<endl;
		DIR *dir;
		struct dirent *ent;
		char fileName[150];
        //string fileName;
        string str1=folder + ".";
        string str2=folder + "..";
        string str3 = folder + "Thumbs.db";        
        //const char folderX[20] = folder;
        char *folderX = const_cast<char*>(folder.c_str());
		
		if ((dir = opendir (folderX)) != NULL) {
		  /* print all the files and directories within directory */
		  while ((ent = readdir (dir)) != NULL) {
			if(flag == '1')
				sprintf (fileName, "OPR-Dataset-New/class1/%s", ent->d_name);
			else if (flag == '2')
				sprintf (fileName, "OPR-Dataset-New/class2/%s", ent->d_name);
			else if(flag == '3')
				sprintf (fileName, "OPR-Dataset-New/class3/%s", ent->d_name);
			else if(flag == '4')
				sprintf (fileName, "OPR-Dataset-New/class4/%s", ent->d_name);
			else if(flag == '5')
				sprintf (fileName, "OPR-Dataset-New/class5/%s", ent->d_name);
			
			if(str1.compare(fileName) == 0 || str2.compare(fileName) == 0 || str3.compare(fileName) == 0)
				continue;
		
			//printf ("%s\n", fileName);
			dataVector.push_back(fileName);
		  }
		  closedir (dir);
		} else {
		  /* could not open directory */
		  perror ("");
		  return EXIT_FAILURE;
		}
		/*
		for(int i=0; i<dataVector.size(); i++) {
			cout<<dataVector.at(i)<<endl;
		}	
		*/
		return 0;
}

int main() {
		srand (time(NULL));
		string folder = "OPR-Dataset-New/class1/";
		string folder1 = "OPR-Dataset-New/class2/";
		string folder2 = "OPR-Dataset-New/class3/";
		string folder3 = "OPR-Dataset-New/class4/";
		string folder4 = "OPR-Dataset-New/class5/";
		setDataVector(folder);
		setDataVector(folder1);
		setDataVector(folder2);
		setDataVector(folder3);
		setDataVector(folder4);
		
		ofstream outputFileTriple;
		outputFileTriple.open("trainingData_New_cvl_triple.txt");
		
		ofstream outputFilePair;
		outputFilePair.open("trainingData_New_cvl_pair.txt");
		
		std::sort(dataVector.begin(), dataVector.end(), compare_by_word);
		for(int i=0; i<dataVector.size(); i++) {
			dataVector.at(i) = "./object_pose_recognition/data/" + dataVector.at(i);
			//cout<<dataVector.at(i)<<endl;
		}	
		cout<<dataVector.size()<<endl;
		
		
		for(int j=0; j<dataVector.size(); j++) {
			if (j % 31 == 0 || j+1>=dataVector.size()) // last frame
				continue;
			else {
				int randPair = rand()%2;
				string pair;
				string triple;
				if((j>=576 && j<=607) || (j>=1184 && j<=1215) || (j>=1792 && j<=1823) || (j>=2400 && j<=2431) || (j>=3008 && j<=3039) ) {
					pair = dataVector.at(j) + " " + dataVector.at(j+1);
				}
				else if(randPair == 0) { // frame-frame+1 (y axis 5 degree);
					pair = dataVector.at(j) + " " + dataVector.at(j+1);
				}
				else if(randPair == 1) {
					pair = dataVector.at(j) + " " + dataVector.at(j+32);
				}
				//cout << pair << endl;
				
				int randTriple = rand()%2;
				
				if(randTriple == 0) { // same class random pose
					if(j>=0 && j<=607) {
						int randSample = rand()%608;
						triple = pair + " " + dataVector.at(randSample);
					}
					else if(j>=608 && j<=1215) {
						int randSample = (rand() % 608) + 608 ;
						triple = pair + " " + dataVector.at(randSample);
					}
					else if(j>=1216 && j<=1823) {
						int randSample = (rand() % 608) + 1216;
						triple = pair + " " + dataVector.at(randSample);
					}
					else if(j>=1824 && j<=2431) {
						int randSample = (rand() % 608) + 1824;
						triple = pair + " " + dataVector.at(randSample);
					}
					else if(j>=2432 && j<=3039) {
						int randSample = (rand() % 608) + 2432;
						triple = pair + " " + dataVector.at(randSample);
					}
				}
				
				else if(randTriple == 1) { // different class random pose
					if(j>=0 && j<=607) {
						int randSample = (rand()%2432)+608;
						triple = pair + " " + dataVector.at(randSample);
 						
					}
					else if(j>=608 && j<=1215) {
						int randDirection = rand()%2;
						if(randDirection == 0) { // take sample from left side
							int randSample = rand()%608;
							triple = pair + " " + dataVector.at(randSample);
						}
						else if(randDirection == 1) { // take sample from right
							int randSample = (rand()%1824) + 1216;
							triple = pair + " " + dataVector.at(randSample);
						}
					}
					else if(j>=1216 && j<=1823) { 
						int randDirection = rand()%2;
						if(randDirection == 0) { //take sample from left
							int randSample = rand()%1216;
							triple = pair + " " + dataVector.at(randSample);
						}
						else if(randDirection == 1) { // from right
							int randSample = (rand()%1216) + 1824;
							triple = pair + " " + dataVector.at(randSample);
						}
					}
					else if(j>=1824 && j<=2431) {
						int randDirection = rand()%2;
						if(randDirection == 0) { //take sample from left
							int randSample = rand()%1824;
							triple = pair + " " + dataVector.at(randSample);
						}
						else if(randDirection == 1) { // from right
							int randSample = (rand()%608) + 2432;
							triple = pair + " " + dataVector.at(randSample);
						}
					}
					else if(j>=2432 && j<=3039) {
						int randSample = rand()%2432;
						triple = pair + " " + dataVector.at(randSample);
					}
				}
				
				triple = triple + "\n";
				pair = pair + "\n";
				//cout<<triple<<endl<<endl;
				outputFileTriple<<triple ;
				outputFilePair<<pair;
				
			}
		}
		
		outputFileTriple.close();
		outputFilePair.close();
		return 0;
}
