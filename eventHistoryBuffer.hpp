#ifndef EVENTHISTORYBUFFER_HPP
#define EVENTHISTORYBUFFER_HPP

#if defined MAX_RES
#else
#define MAX_RES 128
#endif

#include <yarp/sig/all.h>
#include <iCub/emorph/all.h>
#include <vector>
#include <list>

using namespace std;


class eventHistoryBuffer{

private:

    emorph::vtsHelper unwrap;
    double event_time;


public:

    int bufferSize;
    double time_scale;
    int height;
    int width;
    bool disp;
    int x;
    int y;
    int polarity;
    int channel;

    std::list<double> timeStampList[128][128]; //TODO use the global macro definition

    std::list<double>::reverse_iterator timeStampsList_it; //Also should the buffer be initialized to zero

    //Constructor
    eventHistoryBuffer();

    //Destructor
    virtual ~eventHistoryBuffer();

    //Update list at the reference location
    double updateList(emorph::AddressEvent &event);

    //Display list size
    void display();




};



#endif // EVENTHISTORYBUFFER_HPP
