# 1. Red or Blue point

- [listener.py(L462)](https://github.com/junha1125/changsigu/blob/853e333f63349e6e575e095993758cb6315e98b0/211204/listener.py#L461): "parking area 인지를 어떤 색으로 할 것 인가"
- 설정 가능 옵션: (1) `'red'` (2) `'blue'`
- red 일때, 손바닥이 감지되는 문제가 있었다. 'blue'일때 이런 문제가 덜한 듯 하다. (현재는 blue로 해놓은 상태)
- 옵션이 바꿔도 [listener.py(L463~5)](https://github.com/junha1125/changsigu/blob/853e333f63349e6e575e095993758cb6315e98b0/211204/listener.py#L462)에 있는 변수 `front_red_point`, `back_red_point`, `red_minvalue` 는 그대로 사용
- 새로운 마커   
  <img src="https://user-images.githubusercontent.com/46951365/144709845-19fe6c8b-8777-40e0-ad4e-9057e80db433.jpg" alt="drawing" width="400"/>
  
# 2. back_red_point

- [listener.py(L247)](https://github.com/junha1125/changsigu/blob/853e333f63349e6e575e095993758cb6315e98b0/211204/listener.py#L247): `back_red_point` 변수 추가
- `back_red_point` 
  1. 변수는 감지되는 점이 없을 때 = None
  2. 차량 중앙 = \[400, y_valu\]
  3. 차량 오른쪽 = \[500, y_valu\]
  4. 차량 왼쪽 = \[300, y_valu\]
- 따라서 아래와 같이 사용할 것    
  ```python
  if  back_red_point is None:
    - 직진 주행
    - Or 이전 Frame에서 얻은 값 활용
  else:
    if back_red_point[0] > 400:
      - 앞바퀴가 오른쪽을 보도록 (파란화살표)
    else:
      - 앞바퀴가 왼쪽을 보도록 (빨강화살표)
  ```
- 참조 이미지  
  <img src="https://user-images.githubusercontent.com/46951365/144710314-f47c52fb-f55c-464c-a4e9-6cdb03faf4ea.png" alt="drawing" width="400"/>
