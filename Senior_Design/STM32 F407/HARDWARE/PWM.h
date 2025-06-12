#ifndef __PWM_H
#define __PWM_H

#include <sys.h>	  

void TIM3_PWM_Init(u16 arr, u16 psc);
void TIM3_PWM_Duty(float duty);
void TIM3_PWM_Stop(void);
void TIM3_PWM_Start(void);
#endif // __PWM_H
