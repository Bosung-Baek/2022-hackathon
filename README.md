# 2022-hackathon

• 팀명
	GNB

• 제출 세션 및 주제
	특별세션 - 시각 장애인 이동권 침해

• 프로젝트 한 줄 설명
	실시간 객체 탐지 기술과 음성 안내를 통해 시각 장애인의 이동 편의성 증진

• 프로젝트에 대한 설명
	시각 장애인들은 이동 시 많은 어려움이 따른다.
	간단한 외출에도 한두개씩은 있는 계단과 볼라드 등을 인지하기 힘들고
	점자 유도블럭이 없는 경우 횡단보도 이용조차 쉽지 않다.
	버스를 타는 것도 전광판에서 안내 방송이 나오기는 하지만 
	여러 대가 동시에 오는 경우 타려는 버스가 어디에 있는지 알 수 없으므로 이용하기 어렵다.

	이러한 점을 개선하기 위해 카메라를 이용하여 실시간으로 사용자가 보는 방향에서
	이동에 걸림돌이 되는 사물과 위치를 인식한 뒤
	왼쪽에 자동차 있음, 중앙에 사람이 있음 등과 같이 사용자에게 음성안내를 제공해
	시각 장애인들의 이동 편의성을 증진시킨다.

	현재는 단순한 pc용 코드로 구성하였지만 이후 웹캠과 연동하여 휴대폰과 연동하게 된다면
	휴대성도 올라갈 것이며 더욱 세부적인 감지(장애물까지의 거리 파악, 시야각을 고려한 방향)
	를 할 수 있게 될 것으로 기대된다. 

	현재 구현한 기술은 기술적 한계로 인해 단순한 객체 인식 기능만 가능한 수준이지만 이후 카메라를 
	두 개 활용하여 거리 측정이 가능하도록 하고, 단순히 객체(덩어리)가 아닌 기둥이나 바닥의 돌맹이
	와 같은 장애물들도 감지하게끔 모델을 학습시킨다면 더욱 효과적으로 장애물 감지가 가능할
	것으로 기대된다.


• 프로젝트에 활용된 기술
	크게 openCV와 YOLO를 사용하여 객체 감지를 하고자 하였다.
![image](https://user-images.githubusercontent.com/81071956/192131115-d7fa7d04-c0d6-4312-b015-14333125bed7.png)

	시각 장애인을 대상으로 하여 시각적 정보를 컴퓨터가 알아보기 쉽게 하기 위해
	객체의 테두리만 따오도록 코드를 구성하였다. 
	
	또한 tts기능을 기반으로 하여 시각장애인이 음성으로 안내를 받을 수 있게끔 구현하였다.

• 시연영상
	https://youtu.be/hwwwPEDcsZI
