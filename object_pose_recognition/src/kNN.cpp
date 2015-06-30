#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <ctime>
#include <fstream>
#include <cmath>
#include <CL/cl_platform.h>

using namespace std;

char trainingDescriptorsFile[150] = "./object_pose_recognition/data/descriptors.txt";
char testDescriptorsFile[150] = "./object_pose_recognition/data/descriptors_test.txt";

// DataModel class
class DataModel {
	
	private:
		string location;
		int classLabel;
		int curve;
		int rot;
		vector<float> descriptorVect;
	
	public: 
		DataModel(string sLocation, int sClassLabel, int sCurve, int sRot, vector<float> sDescriptorVect) {
			location = sLocation;
			classLabel = sClassLabel;
			curve = sCurve;
			rot = sRot;
			descriptorVect = sDescriptorVect;
		}
		DataModel() {
		}
		void setLocation(string sLocation) {
			this->location = sLocation;
		}
		string getLocation() {
			return this->location;
		}
		
		void setClassLabel(int sClassLabel) {
			this->classLabel = sClassLabel;
		}
		int getClassLabel() {
			return this->classLabel;
		}
		
		void setCurve(int sCurve) {
			this->curve = sCurve;
		}
		int getCurve() {
			return this->curve;
		}
		
		void setRot(int sRot) {
			this->rot = sRot;
		}
		int getRot() {
			return this->rot;
		}
		
		void setDescriptorVect(vector<float> sVect) {
			this->descriptorVect = sVect;
		}
		vector<float> getDescriptorVect() {
			return this->descriptorVect;
		}
	
};

// Model class for difference
class Difference {
      public:
	int classLabel;
	int curve;
	int rot;
	float euclideanDist;
	Difference(int pClassLabel, int pCurve, int pRot, float pEDistance) {
	  classLabel = pClassLabel;
	  curve = pCurve;
	  rot = pRot;
	  euclideanDist = pEDistance;
	}
};

// Result class
class Result {
    
    public:
	int classLabel;
	int frequency;
	Result(int pClassType, int pFreq) {
	    classLabel = pClassType;
	    frequency = pFreq;
	}
	void increaseFrequency() {
	    frequency++;
	}
};

vector<DataModel> trainingDescriptorsVect;
vector<DataModel> testingDescriptorsVect;

// Read training data
void readTrainingDescriptorInformation() {
	
	string line;
	ifstream myReadFile;
	myReadFile.open(trainingDescriptorsFile);
	int z = 1;
	int counter = 0;
	char output[250];
	 if (myReadFile.is_open()) {
		string location;
		int classLabel;
		int curve;
		int rot;
		vector<float> descriptor;
		
		while (!myReadFile.eof()) {
			myReadFile >> output;
			
			if(z==21) {
				DataModel *temp = new DataModel(location, classLabel, curve, rot, descriptor);
				trainingDescriptorsVect.push_back(*temp);
				/*
				cout<<"Location: "<<temp->getLocation()<<endl;
				cout<<"Class Label: "<<temp->getClassLabel()<<endl;
				cout<<"Curve: "<<temp->getCurve()<<endl;
				cout<<"Rot: "<<temp->getRot()<<endl;
				for(int t=0; t<descriptor.size(); t++)
				  cout<<"Descriptor element: "<<descriptor.at(t)<<endl;
				*/
				descriptor.clear();
				z=1;
			}
			
			if(myReadFile.eof())
				break;

			if(z == 1) {
				location = output;
				//cout<<"Location: "<<location<<endl;
				counter++;
			}
			else if(z == 2) {
				sscanf(output, "%d", &classLabel);
				//cout<<"ClassLabel: "<<classLabel<<endl;
			}
			else if(z == 3) {
				sscanf(output, "%d", &curve);
				//cout<<"Curve: "<<curve<<endl;
			}
			else if(z == 4) {
				sscanf(output, "%d", &rot);
				//cout<<"Rot: "<<rot<<endl;
			}
			else if (z>4 && z<21) {
				float temp;
				sscanf(output, "%f", &temp);
				//cout<<"Temp: "<<temp<<endl;
				descriptor.push_back(temp);
			}
			z++;
			//cout<<output<<endl<<endl;
		}
	}
	myReadFile.close();
	
	/*
	for(int y=0; y<trainingDescriptorsVect.size(); y++) {
	    cout<<"Location: "<<trainingDescriptorsVect.at(y).getLocation()<<endl;
	    cout<<"Class Label: "<<trainingDescriptorsVect.at(y).getClassLabel()<<endl;
	    cout<<"Curve: "<<trainingDescriptorsVect.at(y).getCurve()<<endl;
	    cout<<"Rot:"<<trainingDescriptorsVect.at(y).getRot()<<endl;
	    for(int r=0; r<trainingDescriptorsVect.at(y).getDescriptorVect().size(); r++) {
		  cout<<"Descriptor Element: "<<trainingDescriptorsVect.at(y).getDescriptorVect().at(r)<<endl;
	    }
	}
	*/

	//cout<<trainingDescriptorsVect.size()<<endl;

}

// Read testing data
void readTestingDescriptorInformation() {
	
	string line;
	ifstream myReadFile;
	myReadFile.open(testDescriptorsFile);
	int z = 1;
	int counter = 0;
	char output[250];
	 if (myReadFile.is_open()) {
		string location;
		int classLabel;
		int curve;
		int rot;
		vector<float> descriptor;
		
		while (!myReadFile.eof()) {
			myReadFile >> output;
			
			if(z==21) {
				DataModel *temp = new DataModel(location, classLabel, curve, rot, descriptor);
				testingDescriptorsVect.push_back(*temp);
				/*
				cout<<"Location: "<<temp->getLocation()<<endl;
				cout<<"Class Label: "<<temp->getClassLabel()<<endl;
				cout<<"Curve: "<<temp->getCurve()<<endl;
				cout<<"Rot: "<<temp->getRot()<<endl;
				for(int t=0; t<descriptor.size(); t++)
				  cout<<"Descriptor element: "<<descriptor.at(t)<<endl;
				*/
				descriptor.clear();
				z=1;
			}
			
			if(myReadFile.eof())
				break;

			if(z == 1) {
				location = output;
				//cout<<"Location: "<<location<<endl;
				counter++;
			}
			else if(z == 2) {
				sscanf(output, "%d", &classLabel);
				//cout<<"ClassLabel: "<<classLabel<<endl;
			}
			else if(z == 3) {
				sscanf(output, "%d", &curve);
				//cout<<"Curve: "<<curve<<endl;
			}
			else if(z == 4) {
				sscanf(output, "%d", &rot);
				//cout<<"Rot: "<<rot<<endl;
			}
			else if (z>4 && z<21) {
				float temp;
				sscanf(output, "%f", &temp);
				//cout<<"Temp: "<<temp<<endl;
				descriptor.push_back(temp);
			}
			z++;
			//cout<<output<<endl<<endl;
		}
	}
	myReadFile.close();
	
	/*
	for(int y=0; y<testingDescriptorsVect.size(); y++) {
	    cout<<"Location: "<<testingDescriptorsVect.at(y).getLocation()<<endl;
	    cout<<"Class Label: "<<testingDescriptorsVect.at(y).getClassLabel()<<endl;
	    cout<<"Curve: "<<testingDescriptorsVect.at(y).getCurve()<<endl;
	    cout<<"Rot:"<<testingDescriptorsVect.at(y).getRot()<<endl;
	    for(int r=0; r<testingDescriptorsVect.at(y).getDescriptorVect().size(); r++) {
		  cout<<"Descriptor Element: "<<testingDescriptorsVect.at(y).getDescriptorVect().at(r)<<endl;
	    }
	}
	*/

	//cout<<testingDescriptorsVect.size()<<endl;

}
// Sort Difference vector
bool compare_by_dist(const Difference lhs, const Difference rhs) {
    return lhs.euclideanDist < rhs.euclideanDist;
}
// Sort Result vector
bool compare_by_freq(const Result lhs, const Result rhs) {
    return lhs.frequency > rhs.frequency;
}
// k-NN - parameter should be sent from main func
void knnClassify(int k, int pAngleThreshold) {
  
    int correctClassification = 0;
    int correctPoseClassification=0;
    for(int i=0; i<testingDescriptorsVect.size(); i++) {
      vector<Difference> differenceVect;
      differenceVect.clear();
      
      for(int j=0; j<trainingDescriptorsVect.size(); j++) {
	  float dist = 0;
	  for(int z=0; z<testingDescriptorsVect.at(i).getDescriptorVect().size(); z++) {
	    float temp;
	    temp = pow((testingDescriptorsVect.at(i).getDescriptorVect().at(z)-trainingDescriptorsVect.at(j).getDescriptorVect().at(z)),2);
	    dist = dist + temp;
	  }
	  dist = sqrt(dist);
	  Difference *diffTemp = new Difference(trainingDescriptorsVect.at(j).getClassLabel(), trainingDescriptorsVect.at(j).getCurve(), trainingDescriptorsVect.at(j).getRot(), dist);
	  differenceVect.push_back(*diffTemp);
      }
      
      std::sort(differenceVect.begin(), differenceVect.end(), compare_by_dist);
      
      vector<Result> resultVect;
      for(int h=0; h<5; h++) {
	  Result *tempResult = new Result(h+1, 0);
	  resultVect.push_back(*tempResult);
      }
      
      //cout<<"Test label: "<<testingDescriptorsVect.at(i).getClassLabel()<<endl;
      // 1-NN. If you wanna increase k, simply change k value which you send to function as an argument
      for(int p=0; p<k; p++) {
	// update the frequencies according to k'th value
	  if(differenceVect.at(p).classLabel == 1) {
	      resultVect.at(1-1).increaseFrequency();
	  }
	  else if(differenceVect.at(p).classLabel == 2) {
	      resultVect.at(2-1).increaseFrequency();
	  }
	  else if(differenceVect.at(p).classLabel == 3) {
	      resultVect.at(3-1).increaseFrequency();
	  }
	  else if(differenceVect.at(p).classLabel == 4) {
	      resultVect.at(4-1).increaseFrequency();
	  }
	  else if(differenceVect.at(p).classLabel == 5) {
	      resultVect.at(5-1).increaseFrequency();
	  }
      }
      
      // Sort resultVect which has the occurance frequencies of classes so the top elements would be the likely classes
      std::sort(resultVect.begin(), resultVect.end(), compare_by_freq);
      vector<int> likelyClasses;
      int max=0;
      for(int u=0; u<5; u++){
	  if(resultVect.at(u).frequency>=max) {
	      max = resultVect.at(u).frequency;
	      likelyClasses.push_back(resultVect.at(u).classLabel);
	  }
      }
      
      // Object recognition part
      for(int g=0; g<likelyClasses.size(); g++) {
	   if(likelyClasses.at(g)==testingDescriptorsVect.at(i).getClassLabel()) {
	     correctClassification++;
	     break;
	  }
	     
      }
      
      // Pose estimation part
      int curveTotal = 0;
      int rotTotal = 0;
      int poseEstCounter = 0;
      
      for(int y=0; y<likelyClasses.size(); y++) {
	   for(int f=0; f<k; f++) {
		if(differenceVect.at(f).classLabel == likelyClasses.at(y)) {
		      curveTotal = curveTotal + differenceVect.at(f).curve;
		      rotTotal = rotTotal + differenceVect.at(f).rot;
		      poseEstCounter++;
		}
	  }
      }
      
      float curveAvg = (float)curveTotal/poseEstCounter;
      float rotAvg = (float)rotTotal/poseEstCounter;
      
      
      if(abs(testingDescriptorsVect.at(i).getCurve()-curveAvg)<pAngleThreshold && abs(testingDescriptorsVect.at(i).getRot()-rotAvg)<pAngleThreshold) {
	  correctPoseClassification++;
      }
      
      /*
      for(int p=0; p<5; p++) {
	cout<<"Nearest elements: "<<differenceVect.at(p).classLabel<<endl;
       }*/
      resultVect.clear();
      likelyClasses.clear();
    }
    // Calculate the total accuracy of the prediction
    float accuracyObjDet = (float)correctClassification/testingDescriptorsVect.size();
    cout<<"Correct Classification: (Object Recognition) "<<correctClassification<<endl;
    cout<<"Accuracy: (Object Recognition)"<< accuracyObjDet<<endl;
    
    float accuracyPoseEst = (float)correctPoseClassification/testingDescriptorsVect.size();
    cout<<"Correct Classification: (Pose Estimation) "<<correctPoseClassification<<endl;
    cout<<"Accuracy: (Pose Estimation)"<< accuracyPoseEst<<endl;
}

int main() {
	readTrainingDescriptorInformation();
	readTestingDescriptorInformation();
	cout<<"Training Vect size: "<<trainingDescriptorsVect.size()<<endl;
	cout<<"Test Vect size: "<<testingDescriptorsVect.size()<<endl;
	int k=1;
	int angleThreshold = 15;
	knnClassify(k, angleThreshold);
	cout<<k<<"-NN is applied"<<endl;
	cout<<"Finished"<<endl;
	return 0;
}
