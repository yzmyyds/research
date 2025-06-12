/**
  ******************************************************************************
  * File Name          : CAN.h
  * Description        : This file provides code for the configuration
  *                      of the CAN instances.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */
/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __can_H
#define __can_H
#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* USER CODE BEGIN Includes */

#include "stdbool.h"
#define MOTOR_ZERO_POSITION 0  // 假设零位置为0
#define MOTOR_MAX_POSITION 65535  // 最大位置（表示一圈）
typedef struct {
	__IO CAN_RxHeaderTypeDef CAN_RxMsg;
	__IO uint8_t rxData[32];
	
	__IO CAN_TxHeaderTypeDef CAN_TxMsg;
	__IO uint8_t txData[32];

	__IO bool rxFrameFlag;
}CAN_t;
typedef struct {
    int16_t velocity;        // 电机实时转速
    int32_t position;        // 电机实时位置
    uint16_t busVoltage;     // 总线电压 (mV)
    uint16_t current;        // 总线电流 (mA)
    uint16_t encoderValue;   // 编码器值
    uint8_t status;          // 系统状态标志
} MotorStatus_t;
/* USER CODE END Includes */

extern CAN_HandleTypeDef hcan1;
extern MotorStatus_t Motor_Status;
extern uint8_t RxData[8];
/* USER CODE BEGIN Private defines */

extern __IO CAN_t can;

/* USER CODE END Private defines */

void CAN_Init(void);

/* USER CODE BEGIN Prototypes */

void can_SendCmd(__IO uint8_t *cmd, uint8_t len);
/* USER CODE END Prototypes */
void CAN_Receive_Message(uint32_t* id, uint8_t *data, uint8_t* len);
#ifdef __cplusplus
}
#endif
#endif /*__ can_H */

/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
