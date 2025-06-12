#include <stdio.h>
#include "MOTOR.h"
#include "delay.h"
#include "PWM.h"
#include "sys.h"
#include "SENSOR.h"
#include "usart.h"
#include "can.h"
#include "Emm_42.h"
#include "ROS.h"
#include <stdbool.h>

bool grip_cmd=false;          
bool release_cmd=false;
int main(void)
{
	uint8_t data_receive[12];
	uint8_t len;
	uint32_t id;
	typedef enum {
		CLAW_IDLE = 0,
		CLAW_WAIT,
        CLAW_GRIPPING,
        CLAW_HOLDING,
        CLAW_RELEASING,
		GET,
    } CLAW_STATE;
	CLAW_STATE State=CLAW_IDLE;

	// 系统初始化
	Stm32_Clock_Init(336, 8, 2, 7);
	delay_init(168);
	uart_init(115200);
	ADC_Init();
	Sensor_Init();
	CAN_Init();
	//USER_CAN1_Filter_Init();
    // 打开中断接收
	// 初始化 Emm_V5 内部需要的函数（假设 can_SendCmd 调用了 HAL_CAN_AddTxMessage）
	

    Emm_V5_En_Control(0x01, true, false);   // 使能电机
    HAL_Delay(100);

    Emm_V5_Modify_Ctrl_Mode(0x01, false, 2); // 设置为闭环控制模式（模式2）
    HAL_Delay(100);
    
    printf("Init Finish\r\n");
	while(1)
	{
		uint8_t reset=0;
		int32_t pos=0;
		Check_ROS_Command();
		//printf("%d\r\n",grip_cmd);
		switch (State) 
		{
			case CLAW_IDLE :
				Emm_V5_Reset_CurPos_To_Zero(0x01);  // 初始设为0
                printf("State: CLAW_IDLE\r\n");
                State = CLAW_WAIT;  // 等待命令
				HAL_Delay(50);
				break;
			case CLAW_WAIT :
				if (grip_cmd) 
				{
					grip_cmd=false;
					Emm_V5_Pos_Control(0x01, 0, 100, 0, 1000, false, false);
					printf("State: CLAW_GRIPPING\r\n");
					State = CLAW_WAIT;
				}
				else if (release_cmd)
				{
					State = GET;
				}
				HAL_Delay(50);
				break;
			case CLAW_GRIPPING :
				if (Pressure_Trigger())
				{
					Emm_V5_Stop_Now(0x01,false);
					ToROS_Message("GRIP_COMPLETE\r\n");//send complete signal to ROS
                    printf("State: CLAW_HOLDING\r\n");
					printf("Wait for Release cmd ...\r\n"); 
					//State = CLAW_HOLDING;
					State=GET;
				}
				HAL_Delay(50);
				break;
			case GET :
				while (1) {
					Emm_V5_Read_Sys_Params(0x01,S_CPOS);
					CAN_Receive_Message(&id, data_receive, &len);
					if (data_receive[0] == 0x36) // 确保是位置反馈
					{
						// 提取 24 位位置数据
						pos = (data_receive[2] << 24) | (data_receive[3] << 16) | (data_receive[4] << 8)  |  data_receive[5];

						// 根据符号位转换为有符号值
						if (data_receive[1] == 0x01) pos = -pos;
						printf("Current position: %d\r\n", pos);
						break;
					}
					else printf("loop\r\n");
				}
					State=CLAW_HOLDING;
					break;
			case CLAW_HOLDING :
				if (release_cmd)
				{
					release_cmd=false;
					//Emm_V5_Origin_Trigger_Return(0x01, 2, false);
					//Emm_V5_Pos_Control(0x01, 1, 100, 0, pos, false, false); 
					Emm_V5_Pos_Control(0x01, 1, 100, 0, pos, true, false);
					printf("State: CLAW_RELEASING\r\n");
					State=CLAW_RELEASING;
				}
				HAL_Delay(50);
				break;
			case CLAW_RELEASING :
//				while (reset!=1) 
//				{
//					Emm_V5_Read_Sys_Params(0x01,S_CPOS);
//					CAN_Receive_Message(&id, data_receive, &len);
////					if (data_receive[0] == 0x36) // 确保是位置反馈
////					{
////						// 提取 24 位位置数据
////						pos = (data_receive[2] << 24) | (data_receive[3] << 16) | (data_receive[4] << 8)  |  data_receive[5];

////						// 根据符号位转换为有符号值
////						if (data_receive[1] == 0x01) pos = -pos;
////						printf("Current position: %d\r\n", pos);

////						// 判断是否归零（容差范围 ±10）
////						if (pos >= -5 && pos <= 5)
////						{
////							reset=1;
//////							Emm_V5_Stop_Now(0x01,false);
////							printf("Motor has returned to zero.\r\n");
////						}
////					}
//				}
//				//State=CLAW_IDLE;
				State=CLAW_WAIT;
//				//Emm_V5_Reset_CurPos_To_Zero(0x01);
				ToROS_Message("RELEASE_COMPLETE");
				printf("State: CLAW_RELEASED\r\n");
				printf("Wait for next cmd ...\r\n");
				HAL_Delay(50);
				break;
			default :
				State=CLAW_IDLE;
				break;
		}
		HAL_Delay(50);
	}
		
}
void Error_Handler(void)
{
	printf("Error\r\n");
}
