#ifndef _ROS_H
#define _ROS_H
#include "stdbool.h"
#include "sys.h"

void ToROS_Message(char* msg);
void Check_ROS_Command(void);
uint8_t read_uart(void);
#endif
