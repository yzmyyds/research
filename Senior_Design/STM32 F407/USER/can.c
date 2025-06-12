/**
  ******************************************************************************
  * File Name          : CAN.c
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

/* Includes ------------------------------------------------------------------*/
#include "can.h"

/* USER CODE BEGIN 0 */

__IO CAN_t can = {0};

/* USER CODE END 0 */

CAN_HandleTypeDef hcan1;
CAN_FilterTypeDef canFilterConfig;
MotorStatus_t Motor_Status;
uint8_t RxData[8];

void CAN_Init(void)
{
    // Initialize CAN peripheral
    hcan1.Instance = CAN1;
    //hcan.Init.Mode = CAN_MODE_NORMAL; // Normal mode
    hcan1.Init.Mode = CAN_MODE_NORMAL;//回环模式自己发自己收用作测试
    hcan1.Init.Prescaler = 14; // Adjusted for 115200 baud rate
    hcan1.Init.SyncJumpWidth = CAN_SJW_1TQ;
    hcan1.Init.TimeSeg1 = CAN_BS1_4TQ;
    hcan1.Init.TimeSeg2 = CAN_BS2_1TQ;
    hcan1.Init.TimeTriggeredMode = DISABLE;
    hcan1.Init.AutoBusOff = DISABLE;
    hcan1.Init.AutoWakeUp = DISABLE;
    hcan1.Init.AutoRetransmission = DISABLE;
    hcan1.Init.ReceiveFifoLocked = DISABLE;
    hcan1.Init.TransmitFifoPriority = DISABLE;

    if (HAL_CAN_Init(&hcan1) != HAL_OK)
    {
        // Initialization Error
		printf("CAN_Init");
        Error_Handler();
    }

    // Configure CAN filter
    canFilterConfig.FilterBank = 0;
    canFilterConfig.FilterMode = CAN_FILTERMODE_IDMASK;
    canFilterConfig.FilterScale = CAN_FILTERSCALE_32BIT;
    canFilterConfig.FilterIdHigh = 0x0000;
    canFilterConfig.FilterIdLow = 0x0000;
    canFilterConfig.FilterMaskIdHigh = 0x0000;
    canFilterConfig.FilterMaskIdLow = 0x0000;
    canFilterConfig.FilterFIFOAssignment = CAN_RX_FIFO0;
    canFilterConfig.FilterActivation = ENABLE;
    canFilterConfig.SlaveStartFilterBank = 14;

    if (HAL_CAN_ConfigFilter(&hcan1, &canFilterConfig) != HAL_OK)
    {
        // Filter configuration Error
		printf("CAN_Filter");
        Error_Handler();
    }

    // Start CAN
    if (HAL_CAN_Start(&hcan1) != HAL_OK)
    {
        // Start Error
		printf("CAN_Start");
        Error_Handler();
    }

    // Activate CAN RX notification
    if (HAL_CAN_ActivateNotification(&hcan1, CAN_IT_RX_FIFO0_MSG_PENDING) != HAL_OK)
    {
        // Notification Error
		printf("CAN_Act");
        Error_Handler();
    }
}

void HAL_CAN_MspInit(CAN_HandleTypeDef *hcan1)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    if (hcan1->Instance == CAN1)
    {
        __HAL_RCC_CAN1_CLK_ENABLE();      // 开启 CAN1 时钟
        __HAL_RCC_GPIOD_CLK_ENABLE();     // 开启 GPIOD 时钟

        // 配置 PD0（TX）和 PD1（RX）为复用模式 AF9（CAN1）
        GPIO_InitStruct.Pin = GPIO_PIN_0 | GPIO_PIN_1;
        GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
        GPIO_InitStruct.Pull = GPIO_PULLUP;
        GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
        GPIO_InitStruct.Alternate = GPIO_AF9_CAN1;   // ? 设置为 AF9！

        HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);
    }
}

void HAL_CAN_MspDeInit(CAN_HandleTypeDef *hcan1)
{
    if (hcan1->Instance == CAN1)
    {
        // Disable CAN1 clock
        __HAL_RCC_CAN1_CLK_DISABLE();

        // Deinitialize CAN RX and TX pins
        HAL_GPIO_DeInit(GPIOD, GPIO_PIN_0 | GPIO_PIN_1);
    }
}

/**
	* @brief   CAN发送多个字节
	* @param   无
	* @retval  无
	*/
void can_SendCmd(__IO uint8_t *cmd, uint8_t len)
{
	int debug_i;
	
	static uint32_t TxMailbox; __IO uint8_t i = 0, j = 0, k = 0, l = 0, packNum = 0;

	// 除去ID地址和功能码后的数据长度
	j = len - 2;

	// 发送数据
	while(i < j)
	{
		// 数据个数
		k = j - i;

		// 填充缓存
		can.CAN_TxMsg.StdId = 0x00;
		can.CAN_TxMsg.ExtId = ((uint32_t)cmd[0] << 8) | (uint32_t)packNum;
		can.txData[0] = cmd[1];
		can.CAN_TxMsg.IDE = CAN_ID_EXT;
		can.CAN_TxMsg.RTR = CAN_RTR_DATA;

		// 小于8字节命令
		if(k < 8)
		{
			for(l=0; l < k; l++,i++) { can.txData[l + 1] = cmd[i + 2]; } can.CAN_TxMsg.DLC = k + 1;
		}
		// 大于8字节命令，分包发送，每包数据最多发送8个字节
		else
		{
			for(l=0; l < 7; l++,i++) { can.txData[l + 1] = cmd[i + 2]; } can.CAN_TxMsg.DLC = 8;
		}
		
		
		printf("Sending CAN packet %d: ExtId=0x%X, DLC=%d, Data=", packNum, can.CAN_TxMsg.ExtId, can.CAN_TxMsg.DLC);
        for (debug_i = 0; debug_i < can.CAN_TxMsg.DLC; debug_i++) {
            printf("%02X ", can.txData[debug_i]);
        }
        printf("\r\n");
		
		
		// 发送数据
		while(HAL_CAN_AddTxMessage((&hcan1), (CAN_TxHeaderTypeDef *)(&can.CAN_TxMsg), (uint8_t *)(&can.txData), (&TxMailbox)) != HAL_OK);

		// 记录发送的第几包的数据
		++packNum;
	}
}

/* USER CODE END 1 */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
//void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hcan1)
//{
//    CAN_RxHeaderTypeDef rxHeader;
//    uint8_t rxData[8];

//    // 获取接收到的CAN消息
//    if (HAL_CAN_GetRxMessage(hcan1, CAN_RX_FIFO0, &rxHeader, rxData) == HAL_OK)
//    {
//        // 根据返回的地址和功能码来解析数据
//		printf("Raw CAN: %02X %02X %02X %02X %02X %02X %02X %02X\r\n",
//       rxData[0], rxData[1], rxData[2], rxData[3],
//       rxData[4], rxData[5], rxData[6], rxData[7]);
//        if (rxData[1] == 0x35)  // 实时转速响应
//        {
//            // 解析电机实时转速
//            if (rxData[2] == 0x01) {
//                Motor_Status.velocity = (int16_t)((rxData[4] << 8) | rxData[3]);
//                Motor_Status.velocity = -Motor_Status.velocity;  // 转速方向
//            } else {
//                Motor_Status.velocity = (int16_t)((rxData[4] << 8) | rxData[3]);
//            }
//            printf("Motor Speed: %d RPM\r\n", Motor_Status.velocity);
//        }
//        else if (rxData[1] == 0x36)  // 实时位置响应
//        {
//            // 解析电机实时位置
//            if (rxData[2] == 0x01) {
//                Motor_Status.position = ((int32_t)rxData[6] << 24) | ((int32_t)rxData[5] << 16) |
//                                        ((int32_t)rxData[4] << 8) | rxData[3];
//                Motor_Status.position = -Motor_Status.position;  // 位置方向
//            } else {
//                Motor_Status.position = ((int32_t)rxData[6] << 24) | ((int32_t)rxData[5] << 16) |
//                                        ((int32_t)rxData[4] << 8) | rxData[3];
//            }
//            printf("Motor Position: %d POS\r\n", Motor_Status.position);
//        }
//        else if (rxData[1] == 0x43)  // 系统状态响应
//        {
//            // 解析系统状态参数
//            Motor_Status.busVoltage = (rxData[2] << 8) | rxData[3];
//            Motor_Status.current = (rxData[4] << 8) | rxData[5];
//            Motor_Status.encoderValue = (rxData[6] << 8) | rxData[7];
//            printf("Bus Voltage: %d mV, Current: %d mA, Encoder Value: %d\r\n",
//                   Motor_Status.busVoltage, Motor_Status.current, Motor_Status.encoderValue);
//        }
//        else {
//			
//            printf("Unknown CAN message received.\r\n");
//        }
//    }
//}


void CAN_Receive_Message(uint32_t* id, uint8_t *data, uint8_t* len)
{
//    // 接收CAN数据包
//    CAN_RxHeaderTypeDef RxHeader;
//	uint8_t i;
//	
//	RxHeader.IDE=CAN_ID_STD;
////	int i;
////    
////    // 读取CAN消息
////	printf("Receive Start Success");
////    if (HAL_CAN_GetRxFifoFillLevel(&hcan1, CAN_RX_FIFO0) > 0) {
////        HAL_CAN_GetRxMessage(&hcan1, CAN_RX_FIFO0, &RxHeader, RxData);
////        printf("Raw Data: ");
////		for (i = 0; i < RxHeader.DLC; i++) {
////			printf("%02X ", RxData[i]);
////		}
////		printf("\n");
////        // 检查是否为返回的电机参数
////        if (RxData[0] == 0x01 && RxData[1] == 0xF3) {  // 电机地址为0x01，功能码为0xF3
////            int32_t position = (int32_t)(RxData[2] << 24 | RxData[3] << 16 | RxData[4] << 8 | RxData[5]);
////            Motor_Status.position = position; // 解析位置并存储
////            printf("Motor Current Position: %ld\r\n", (long)Motor_Status.position);
////		}
////    }

//	
//    // Receive the message
//    if (HAL_CAN_GetRxMessage(&hcan1, CAN_RX_FIFO0, &RxHeader, RxData) != HAL_OK)
//    {
//        // Reception Error
//        Error_Handler();
//    }

//    // Get the message ID and length

//    printf("Receive info:\r\n");//将buf中的数字打印出来
//    for(i = 0;i <8;i++)
//    {
//        printf("%X ",RxData[i]);
//    }
//	printf("\r\n");
	CAN_RxHeaderTypeDef rxHeader;
	uint8_t i =0;
	
    // Receive the message
    if (HAL_CAN_GetRxMessage(&hcan1, CAN_RX_FIFO0, &rxHeader, data) != HAL_OK)
    {
        // Reception Error
        Error_Handler();
    }

    // Get the message ID and length
    *id = rxHeader.StdId;
    *len = rxHeader.DLC;

    printf("Receive info:\r\n");//将buf中的数字打印出来
    for(i = 0;i <*len;i++)
    {
        printf("%X ",data[i]);
    }
	printf("\r\n");
}



















///**
//  ******************************************************************************
//  * File Name          : CAN.c
//  * Description        : This file provides code for the configuration
//  *                      of the CAN instances.
//  ******************************************************************************
//  * @attention
//  *
//  * <h2><center>&copy; Copyright (c) 2024 STMicroelectronics.
//  * All rights reserved.</center></h2>
//  *
//  * This software component is licensed by ST under BSD 3-Clause license,
//  * the "License"; You may not use this file except in compliance with the
//  * License. You may obtain a copy of the License at:
//  *                        opensource.org/licenses/BSD-3-Clause
//  *
//  ******************************************************************************
//  */

///* Includes ------------------------------------------------------------------*/
//#include "can.h"

///* USER CODE BEGIN 0 */

//__IO CAN_t can = {0};

///* USER CODE END 0 */

//CAN_HandleTypeDef hcan2;
//CAN_FilterTypeDef canFilterConfig;
//MotorStatus_t Motor_Status;
//uint8_t RxData[8];

//void CAN_Init(void)
//{
//    // Initialize CAN peripheral
//    hcan2.Instance = CAN2;
//    //hcan.Init.Mode = CAN_MODE_NORMAL; // Normal mode
//    hcan2.Init.Mode = CAN_MODE_NORMAL;//回环模式自己发自己收用作测试
//    hcan2.Init.Prescaler = 14; // Adjusted for 115200 baud rate
//    hcan2.Init.SyncJumpWidth = CAN_SJW_1TQ;
//    hcan2.Init.TimeSeg1 = CAN_BS1_4TQ;
//    hcan2.Init.TimeSeg2 = CAN_BS2_1TQ;
//    hcan2.Init.TimeTriggeredMode = DISABLE;
//    hcan2.Init.AutoBusOff = DISABLE;
//    hcan2.Init.AutoWakeUp = DISABLE;
//    hcan2.Init.AutoRetransmission = DISABLE;
//    hcan2.Init.ReceiveFifoLocked = DISABLE;
//    hcan2.Init.TransmitFifoPriority = DISABLE;

//    if (HAL_CAN_Init(&hcan2) != HAL_OK)
//    {
//        // Initialization Error
//		printf("CAN_Init");
//        Error_Handler();
//    }

//    // Configure CAN filter
//    canFilterConfig.FilterBank = 14;
//    canFilterConfig.FilterMode = CAN_FILTERMODE_IDMASK;
//    canFilterConfig.FilterScale = CAN_FILTERSCALE_32BIT;
//    canFilterConfig.FilterIdHigh = 0x0000;
//    canFilterConfig.FilterIdLow = 0x0000;
//    canFilterConfig.FilterMaskIdHigh = 0x0000;
//    canFilterConfig.FilterMaskIdLow = 0x0000;
//    canFilterConfig.FilterFIFOAssignment = CAN_RX_FIFO0;
//    canFilterConfig.FilterActivation = ENABLE;
//    canFilterConfig.SlaveStartFilterBank = 14;

//    if (HAL_CAN_ConfigFilter(&hcan2, &canFilterConfig) != HAL_OK)
//    {
//        // Filter configuration Error
//		printf("CAN_Filter");
//        Error_Handler();
//    }

//    // Start CAN
//    if (HAL_CAN_Start(&hcan2) != HAL_OK)
//    {
//        // Start Error
//		printf("CAN_Start");
//        Error_Handler();
//    }

//    // Activate CAN RX notification
//    if (HAL_CAN_ActivateNotification(&hcan2, CAN_IT_RX_FIFO0_MSG_PENDING) != HAL_OK)
//    {
//        // Notification Error
//		printf("CAN_Act");
//        Error_Handler();
//    }
//}

//void HAL_CAN_MspInit(CAN_HandleTypeDef *hcan2)
//{
//    GPIO_InitTypeDef GPIO_InitStruct = {0};
//    if (hcan2->Instance == CAN2)
//    {
//        __HAL_RCC_CAN1_CLK_ENABLE();      // 开启 CAN1 时钟
//		__HAL_RCC_CAN2_CLK_ENABLE();
//        __HAL_RCC_GPIOB_CLK_ENABLE();     // 开启 GPIOD 时钟

//        // 配置 PD0（TX）和 PD1（RX）为复用模式 AF9（CAN1）
//        GPIO_InitStruct.Pin = GPIO_PIN_12 | GPIO_PIN_13;
//        GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
//        GPIO_InitStruct.Pull = GPIO_PULLUP;
//        GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
//        GPIO_InitStruct.Alternate = GPIO_AF9_CAN2;   // ? 设置为 AF9！

//        HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);
//    }
//}

//void HAL_CAN_MspDeInit(CAN_HandleTypeDef *hcan2)
//{
//    if (hcan2->Instance == CAN1)
//    {
//        // Disable CAN1 clock
//        __HAL_RCC_CAN1_CLK_DISABLE();
//		__HAL_RCC_CAN2_CLK_DISABLE();

//        // Deinitialize CAN RX and TX pins
//        HAL_GPIO_DeInit(GPIOB, GPIO_PIN_12 | GPIO_PIN_13);
//    }
//}

///**
//	* @brief   CAN发送多个字节
//	* @param   无
//	* @retval  无
//	*/
//void can_SendCmd(__IO uint8_t *cmd, uint8_t len)
//{
//	int debug_i;
//	
//	static uint32_t TxMailbox; __IO uint8_t i = 0, j = 0, k = 0, l = 0, packNum = 0;

//	// 除去ID地址和功能码后的数据长度
//	j = len - 2;

//	// 发送数据
//	while(i < j)
//	{
//		// 数据个数
//		k = j - i;

//		// 填充缓存
//		can.CAN_TxMsg.StdId = 0x00;
//		can.CAN_TxMsg.ExtId = ((uint32_t)cmd[0] << 8) | (uint32_t)packNum;
//		can.txData[0] = cmd[1];
//		can.CAN_TxMsg.IDE = CAN_ID_EXT;
//		can.CAN_TxMsg.RTR = CAN_RTR_DATA;

//		// 小于8字节命令
//		if(k < 8)
//		{
//			for(l=0; l < k; l++,i++) { can.txData[l + 1] = cmd[i + 2]; } can.CAN_TxMsg.DLC = k + 1;
//		}
//		// 大于8字节命令，分包发送，每包数据最多发送8个字节
//		else
//		{
//			for(l=0; l < 7; l++,i++) { can.txData[l + 1] = cmd[i + 2]; } can.CAN_TxMsg.DLC = 8;
//		}
//		
//		
//		printf("Sending CAN packet %d: ExtId=0x%X, DLC=%d, Data=", packNum, can.CAN_TxMsg.ExtId, can.CAN_TxMsg.DLC);
//        for (debug_i = 0; debug_i < can.CAN_TxMsg.DLC; debug_i++) {
//            printf("%02X ", can.txData[debug_i]);
//        }
//        printf("\r\n");
//		
//		
//		// 发送数据
//		while(HAL_CAN_AddTxMessage((&hcan2), (CAN_TxHeaderTypeDef *)(&can.CAN_TxMsg), (uint8_t *)(&can.txData), (&TxMailbox)) != HAL_OK);

//		// 记录发送的第几包的数据
//		++packNum;
//	}
//}

///* USER CODE END 1 */

///************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
////void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hcan1)
////{
////    CAN_RxHeaderTypeDef rxHeader;
////    uint8_t rxData[8];

////    // 获取接收到的CAN消息
////    if (HAL_CAN_GetRxMessage(hcan1, CAN_RX_FIFO0, &rxHeader, rxData) == HAL_OK)
////    {
////        // 根据返回的地址和功能码来解析数据
////		printf("Raw CAN: %02X %02X %02X %02X %02X %02X %02X %02X\r\n",
////       rxData[0], rxData[1], rxData[2], rxData[3],
////       rxData[4], rxData[5], rxData[6], rxData[7]);
////        if (rxData[1] == 0x35)  // 实时转速响应
////        {
////            // 解析电机实时转速
////            if (rxData[2] == 0x01) {
////                Motor_Status.velocity = (int16_t)((rxData[4] << 8) | rxData[3]);
////                Motor_Status.velocity = -Motor_Status.velocity;  // 转速方向
////            } else {
////                Motor_Status.velocity = (int16_t)((rxData[4] << 8) | rxData[3]);
////            }
////            printf("Motor Speed: %d RPM\r\n", Motor_Status.velocity);
////        }
////        else if (rxData[1] == 0x36)  // 实时位置响应
////        {
////            // 解析电机实时位置
////            if (rxData[2] == 0x01) {
////                Motor_Status.position = ((int32_t)rxData[6] << 24) | ((int32_t)rxData[5] << 16) |
////                                        ((int32_t)rxData[4] << 8) | rxData[3];
////                Motor_Status.position = -Motor_Status.position;  // 位置方向
////            } else {
////                Motor_Status.position = ((int32_t)rxData[6] << 24) | ((int32_t)rxData[5] << 16) |
////                                        ((int32_t)rxData[4] << 8) | rxData[3];
////            }
////            printf("Motor Position: %d POS\r\n", Motor_Status.position);
////        }
////        else if (rxData[1] == 0x43)  // 系统状态响应
////        {
////            // 解析系统状态参数
////            Motor_Status.busVoltage = (rxData[2] << 8) | rxData[3];
////            Motor_Status.current = (rxData[4] << 8) | rxData[5];
////            Motor_Status.encoderValue = (rxData[6] << 8) | rxData[7];
////            printf("Bus Voltage: %d mV, Current: %d mA, Encoder Value: %d\r\n",
////                   Motor_Status.busVoltage, Motor_Status.current, Motor_Status.encoderValue);
////        }
////        else {
////			
////            printf("Unknown CAN message received.\r\n");
////        }
////    }
////}


//void CAN_Receive_Message(uint32_t* id, uint8_t *data, uint8_t* len)
//{
////    // 接收CAN数据包
////    CAN_RxHeaderTypeDef RxHeader;
////	uint8_t i;
////	
////	RxHeader.IDE=CAN_ID_STD;
//////	int i;
//////    
//////    // 读取CAN消息
//////	printf("Receive Start Success");
//////    if (HAL_CAN_GetRxFifoFillLevel(&hcan1, CAN_RX_FIFO0) > 0) {
//////        HAL_CAN_GetRxMessage(&hcan1, CAN_RX_FIFO0, &RxHeader, RxData);
//////        printf("Raw Data: ");
//////		for (i = 0; i < RxHeader.DLC; i++) {
//////			printf("%02X ", RxData[i]);
//////		}
//////		printf("\n");
//////        // 检查是否为返回的电机参数
//////        if (RxData[0] == 0x01 && RxData[1] == 0xF3) {  // 电机地址为0x01，功能码为0xF3
//////            int32_t position = (int32_t)(RxData[2] << 24 | RxData[3] << 16 | RxData[4] << 8 | RxData[5]);
//////            Motor_Status.position = position; // 解析位置并存储
//////            printf("Motor Current Position: %ld\r\n", (long)Motor_Status.position);
//////		}
//////    }

////	
////    // Receive the message
////    if (HAL_CAN_GetRxMessage(&hcan1, CAN_RX_FIFO0, &RxHeader, RxData) != HAL_OK)
////    {
////        // Reception Error
////        Error_Handler();
////    }

////    // Get the message ID and length

////    printf("Receive info:\r\n");//将buf中的数字打印出来
////    for(i = 0;i <8;i++)
////    {
////        printf("%X ",RxData[i]);
////    }
////	printf("\r\n");
//	CAN_RxHeaderTypeDef rxHeader;
//	uint8_t i =0;
//	
//    // Receive the message
//    if (HAL_CAN_GetRxMessage(&hcan2, CAN_RX_FIFO0, &rxHeader, data) != HAL_OK)
//    {
//        // Reception Error
//        Error_Handler();
//    }

//    // Get the message ID and length
//    *id = rxHeader.StdId;
//    *len = rxHeader.DLC;

//    printf("Receive info:\r\n");//将buf中的数字打印出来
//    for(i = 0;i <*len;i++)
//    {
//        printf("%X ",data[i]);
//    }
//	printf("\r\n");
//}
