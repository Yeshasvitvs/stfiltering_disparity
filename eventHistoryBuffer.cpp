#include "eventHistoryBuffer.hpp"

//Constructor Definition
eventHistoryBuffer::eventHistoryBuffer(){ //one time initialization

    std::cout << "Event History Buffer Constructor..." << std::endl;
    height = MAX_RES;
    width = MAX_RES;
    bufferSize = 50;
    time_scale = 1000000;
    disp = false;



}


double eventHistoryBuffer::updateList(emorph::AddressEvent &event){


    //std::cout<< "Updating the List..." <<std::endl;//Debug Code
    //NOTE : The Sensor X and Y are reversed
    x = event.getY();
    y = event.getX();

    if(x < 0 || x > width-1 || y < 0 || y > height-1) return -1;

    //Get the event timestamp
    event_time = unwrap(event.getStamp());


    //converting to seconds
    event_time = event_time / time_scale;

    //Encoding channel information of the event in timestamp so that both Left and Right eyes events can be stored
    channel = event.getChannel();
    if(channel == 0){channel = -1;} //Changing Left eye channel value to -1
    event_time = channel * event_time;

    //Pushing the event time stamp into the list
    timeStampList[x][y].push_back(event_time);

    //Check if the list size is more than buffer size
    if( timeStampList[x][y].size() > bufferSize){

        timeStampList[x][y].pop_front(); //if list is more than buffer size reduce it to buffer size
    }


    //Call the display
    if(disp==true){ display(); }


    return 0;

}

void eventHistoryBuffer::display(){


    //Checking the event history buffer size
    std::cout << "Stored Buffer Size : " <<  timeStampList[x][y].size() << std::endl;//Debug Code

    if(!timeStampList[x][y].empty()){

        std::cout << "Stored Buffer value : " << timeStampList[x][y].back() << std::endl; //Debug Code

    }

}

//Destructor
eventHistoryBuffer::~eventHistoryBuffer(){

}
