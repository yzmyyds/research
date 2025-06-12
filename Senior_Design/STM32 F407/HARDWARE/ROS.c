#include "ROS.h"
#include "usart.h"
#include <string.h>
#include <stdio.h>
#include <stdbool.h>

extern u16 USART_RX_STA;
extern u8 USART_RX_BUF[USART_REC_LEN];
extern bool grip_cmd;
extern bool release_cmd;

void Check_ROS_Command(void) {
	char* pos;
    if (USART_RX_STA & 0x8000)  // 接收完成
    {
        USART_RX_BUF[USART_RX_STA & 0x3FFF] = '\0';  // 添加字符串结束符
		
        // 去除尾部换行符
        if ((pos = strchr((char*)USART_RX_BUF, '\r')) != NULL) *pos = '\0';
        if ((pos = strchr((char*)USART_RX_BUF, '\n')) != NULL) *pos = '\0';

        if (strcmp((char*)USART_RX_BUF, "S") == 0)
        {
            grip_cmd = true;
            release_cmd = false;
			printf("Receive START Command\r\n");  //test
        }
        else if (strcmp((char*)USART_RX_BUF, "R") == 0)
        {
            grip_cmd = false;
            release_cmd = true;
			printf("Receive RELEASE Command\r\n");
        }
        else if (strcmp((char*)USART_RX_BUF, "H") == 0)
        {
            grip_cmd = false;
            release_cmd = false;
			printf("Receive STOP Command\r\n");
        }

        USART_RX_STA = 0;
        memset(USART_RX_BUF, 0, sizeof(USART_RX_BUF));
    }
}
//u8 read_uart(void)
//{
//	char received_data[USART_REC_LEN] = {0};
//	char *end_ptr;
//    if (USART_RX_STA & 0x8000) {  // 检查接收状态
//        USART_RX_STA &= ~0x8000;  // 清除接收状态
//        // 去除接收到的数据中的 \r\n
//        strncpy(received_data, (char *)USART_RX_BUF, USART_REC_LEN - 1);
//        received_data[USART_REC_LEN - 1] = '\0';  // 保证字符串终止
//        end_ptr = strstr(received_data, "\r\n");
//        if (end_ptr) {
//            *end_ptr = '\0';  // 截断字符串，去除 \r\n
//        }
//        if (strcmp(received_data, "S") == 0) {
//            // 如果接收到 "S" 字符串，返回 1
//			printf("Received data: %s\r\n", received_data);  // 打印接收到的数据
//            return 1;
//        } else  {
//            // 如果接收到 "ERROR" 字符串，返回 0
//			printf("Received data: %s\r\n", received_data);  // 打印接收到的数据
//            return 0;
//        }
//    }
//    // 这里可以根据实际需要实现 UART 数据读取
//	return 2;
//}
void ToROS_Message(char* msg) {
	printf("%s\r\n",msg);
}
