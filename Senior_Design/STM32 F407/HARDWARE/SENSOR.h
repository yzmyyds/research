#ifndef __SENSOR_
#define __SENSOR_

#include <sys.h>

void Sensor_Init(void);
float Sensor_Read(void);
u8 Grasp_Release(void);
void ADC_Init(void);
float Pressure_Read(void);
u8 Pressure_Trigger(void);
#endif

