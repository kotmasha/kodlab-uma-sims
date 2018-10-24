g++ -fPIC -std=c++11 -c Agent.cpp -o Agent.o;
g++ -fPIC -std=c++11 -c logging.cpp -o logging.o;
g++ -fPIC -std=c++11 -c AgentDM.cpp -o AgentDM.o;
swig -c++ -python -o UMA_NEW_wrap.cpp UMA_NEW.i;
gcc -fPIC -c UMA_NEW_wrap.cpp -o UMA_NEW_wrap.o -I/usr/include/python2.7;
nvcc -shared -Xcompiler -fPIC -std=c++11 kernel.cu Agent.o logging.o AgentDM.o UMA_NEW_wrap.o -o _UMA_NEW.so;