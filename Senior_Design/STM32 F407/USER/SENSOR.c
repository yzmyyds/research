#include "SENSOR.h"
#include "delay.h"
#include "stm32f4xx_hal.h"

GPIO_PinState bitstatus;
GPIO_PinState last_state=GPIO_PIN_SET;
ADC_HandleTypeDef hadc3;
void Sensor_Init() {
	GPIO_InitTypeDef GPIO_InitStruct;
	
	__HAL_RCC_GPIOA_CLK_ENABLE();
	GPIO_InitStruct.Pin=GPIO_PIN_0;            //PA0
    GPIO_InitStruct.Mode=GPIO_MODE_INPUT;      //输入
    GPIO_InitStruct.Pull=GPIO_PULLDOWN;        //下拉
    GPIO_InitStruct.Speed=GPIO_SPEED_HIGH;     //高速
    HAL_GPIO_Init(GPIOA,&GPIO_InitStruct);
    
	__HAL_RCC_GPIOE_CLK_ENABLE(); 
	GPIO_InitStruct.Pin=GPIO_PIN_4; //PE2,3,4
    GPIO_InitStruct.Mode=GPIO_MODE_INPUT;      //输入
    GPIO_InitStruct.Pull=GPIO_PULLDOWN;        //下拉
    GPIO_InitStruct.Speed=GPIO_SPEED_HIGH;     //高速
    HAL_GPIO_Init(GPIOE,&GPIO_InitStruct);
	
	__HAL_RCC_GPIOF_CLK_ENABLE(); // 启用 GPIOF 时钟
    GPIO_InitStruct.Pin = GPIO_PIN_6;          //PF6
    GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOF, &GPIO_InitStruct);
	
	GPIO_InitStruct.Pin = GPIO_PIN_3;          //PF6
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    HAL_GPIO_Init(GPIOF, &GPIO_InitStruct);
}


float Sensor_Read() {
	float pressure=0.0;
    bitstatus=HAL_GPIO_ReadPin(GPIOA,GPIO_PIN_0);
    if (bitstatus==GPIO_PIN_SET) {
		pressure=60;
	}
	else pressure=0;
	return pressure;
	
}

u8 Grasp_Release() {
	u8 key_pressed=0;
	GPIO_PinState current_state=HAL_GPIO_ReadPin(GPIOE,GPIO_PIN_4);
	if (current_state == GPIO_PIN_RESET && last_state == GPIO_PIN_SET) {
	    HAL_Delay(50); // 消抖延时
        current_state = HAL_GPIO_ReadPin(GPIOE, GPIO_PIN_4);
        if (current_state == GPIO_PIN_RESET) {
            // 等待按键释放
            while (HAL_GPIO_ReadPin(GPIOE, GPIO_PIN_4) == GPIO_PIN_SET);
            HAL_Delay(50); // 再次消抖
            key_pressed = 1; // 标记有效按键动作
        }
	}
	last_state=current_state;
	return key_pressed;
} 

void ADC_Init(void) {
    ADC_ChannelConfTypeDef sConfig;
    // 配置 PF6 为模拟输入
    __HAL_RCC_ADC3_CLK_ENABLE();  // 启用 ADC3 时钟

    // 配置 ADC3 参数
    hadc3.Instance = ADC3;
    hadc3.Init.Resolution = ADC_RESOLUTION_12B;  // 12 位分辨率
    hadc3.Init.ContinuousConvMode = DISABLE;     // 不使用连续转换模式
    hadc3.Init.DiscontinuousConvMode = DISABLE;
    hadc3.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
    hadc3.Init.DataAlign = ADC_DATAALIGN_RIGHT;
    hadc3.Init.NbrOfConversion = 1;

    HAL_ADC_Init(&hadc3);

    // 配置 ADC3 通道
    sConfig.Channel = ADC_CHANNEL_4;  // 使用 ADC3_IN4（PF6）
    sConfig.SamplingTime = ADC_SAMPLETIME_3CYCLES;  // 采样时间为 3 个时钟周期
    HAL_ADC_ConfigChannel(&hadc3, &sConfig);
}

float Pressure_Read() {
	HAL_ADC_Start(&hadc3);
    if (HAL_ADC_PollForConversion(&hadc3, 10) == HAL_OK) {
        // 获取 ADC 转换结果
        uint32_t adc_value = HAL_ADC_GetValue(&hadc3);
        // 将 ADC 值转换为压力值（假设压力传感器的输出为 0-3.3V，映射到0-100压力值）
        float pressure = (1-(adc_value / 4095.0f)) * 100.0f;
        return pressure;
    }
    return 0.0f;
}
u8 Pressure_Trigger() {
	if (HAL_GPIO_ReadPin(GPIOF, GPIO_PIN_3) == GPIO_PIN_RESET)
        {
            printf("压力超过阈值\r\n");
			return 1;
        }
    else
        {
            printf("压力正常\r\n");
			return 0;
        }
}
