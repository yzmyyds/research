#include "PWM.h"
#include "delay.h"
#include "sys.h"

#define MOTOR_DIR_PIN   GPIO_PIN_1 //PB1 DIR
#define MOTOR_ENA_PIN   GPIO_PIN_7 //PA7 ENA
#define MOTOR_PUL_PIN   GPIO_PIN_0 //PB0 STP
#define MOTOR_DIR_PORT  GPIOB
#define MOTOR_ENA_PORT  GPIOA
#define MOTOR_PUL_PORT  GPIOB

TIM_HandleTypeDef TIM3_Handler;
TIM_OC_InitTypeDef TIM3_CH3_Handler;

void TIM3_PWM_Init(u16 arr, u16 psc) {
	GPIO_InitTypeDef GPIO_InitStruct;
	__HAL_RCC_TIM3_CLK_ENABLE();
	
    TIM3_Handler.Instance=TIM3;
	TIM3_Handler.Init.Prescaler=psc;
	TIM3_Handler.Init.CounterMode=TIM_COUNTERMODE_UP;
	TIM3_Handler.Init.Period=arr;
	TIM3_Handler.Init.ClockDivision=TIM_CLOCKDIVISION_DIV1;
	HAL_TIM_PWM_Init(&TIM3_Handler);
	
	TIM3_CH3_Handler.OCMode = TIM_OCMODE_PWM1;
    TIM3_CH3_Handler.Pulse = 0;  // Initial duty cycle (0)
    TIM3_CH3_Handler.OCPolarity = TIM_OCPOLARITY_HIGH;
    TIM3_CH3_Handler.OCFastMode = TIM_OCFAST_DISABLE;
    HAL_TIM_PWM_ConfigChannel(&TIM3_Handler, &TIM3_CH3_Handler, TIM_CHANNEL_3);
	
    __HAL_RCC_GPIOB_CLK_ENABLE();  // Enable GPIOB clock
    GPIO_InitStruct.Pin = MOTOR_PUL_PIN;  // Configure PB0
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;  // Alternate function push-pull
    GPIO_InitStruct.Pull = GPIO_NOPULL;  // No pull-up or pull-down
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;  // High speed
	GPIO_InitStruct.Alternate = GPIO_AF2_TIM3; 
    HAL_GPIO_Init(MOTOR_PUL_PORT, &GPIO_InitStruct);  // Initialize PB0
	
	//HAL_TIM_PWM_Start(&TIM3_Handler, TIM_CHANNEL_3);
}

void TIM3_PWM_Duty(float duty){
	 TIM3->CCR3 = duty;  // Update the PWM duty cycle (pulses per second) for TIM3 CH3 (PB0)
}

void TIM3_PWM_Stop() {
     HAL_TIM_PWM_Stop(&TIM3_Handler, TIM_CHANNEL_3);	
}

void TIM3_PWM_Start() {
     HAL_TIM_PWM_Start(&TIM3_Handler,TIM_CHANNEL_3);	
}
