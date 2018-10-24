#ifndef _AGENTDM_
#define _AGENTDM_
#include <fstream>
#include "Agent.h"

class Agent;

class AgentDM{
protected:
	Agent *agent;
public:
	AgentDM(Agent *agent);
	~AgentDM();

	bool writeData(string filename);
	bool readData(string filename);
};

#endif