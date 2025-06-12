#include "delay.h"
#include "PWM.h"
#include "sys.h"

#define MOTOR_DIR_PIN   GPIO_PIN_1 //PB1 DIR
#define MOTOR_ENA_PIN   GPIO_PIN_7 //PA7 ENA
#define MOTOR_PUL_PIN   GPIO_PIN_0 //PB0 STP
#define MOTOR_DIR_PORT  GPIOB
#define MOTOR_ENA_PORT  GPIOA
#define MOTOR_PUL_PORT  GPIOB



void MOTOR_Init() {
	GPIO_InitTypeDef GPIO_InitStruct;

    // Enable GPIO clock
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();

//    // Initialize PUL pin (PB0)
//    GPIO_InitStruct.Pin = MOTOR_PUL_PIN;
//    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP; // Alternate function push-pull (for PWM)
//    GPIO_InitStruct.Pull = GPIO_NOPULL;
//    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
//    HAL_GPIO_Init(MOTOR_PUL_PORT, &GPIO_InitStruct);

    // Initialize DIR pin (PB1)
    GPIO_InitStruct.Pin = MOTOR_DIR_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(MOTOR_DIR_PORT, &GPIO_InitStruct);

    // Initialize ENA pin (PA7)
    GPIO_InitStruct.Pin = MOTOR_ENA_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(MOTOR_ENA_PORT, &GPIO_InitStruct);
	
}	

void MOTOR_Dir(u8 dir) {
	if (dir) HAL_GPIO_WritePin(MOTOR_DIR_PORT,MOTOR_DIR_PIN,GPIO_PIN_SET);
	else HAL_GPIO_WritePin(MOTOR_DIR_PORT,MOTOR_DIR_PIN,GPIO_PIN_RESET);
	
}

void MOTOR_Ena(u8 ena) {
    if (ena) {
        HAL_GPIO_WritePin(MOTOR_ENA_PORT, MOTOR_ENA_PIN, GPIO_PIN_SET);
		TIM3_PWM_Start();
    } else {
        HAL_GPIO_WritePin(MOTOR_ENA_PORT, MOTOR_ENA_PIN, GPIO_PIN_RESET);
        TIM3_PWM_Stop();   // ֹͣPWM
    }
}
