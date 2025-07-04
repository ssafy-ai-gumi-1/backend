카테고리: 운영체제
키워드: [페이징, 세그먼테이션, 메모리 관리, 연속 메모리, 불연속 메모리, 프레임, 페이지, 세그먼트, 단편화, 내부 단편화, 외부 단편화, 고정 분할, 동적 분할, 가상 메모리, 메모리 오버헤드]

### 페이징과 세그먼테이션

---

##### 기법을 쓰는 이유

> 다중 프로그래밍 시스템에 여러 프로세스를 수용하기 위해 주기억장치를 동적 분할하는 메모리 관리 작업이 필요해서

<br>

#### 메모리 관리 기법

1. 연속 메모리 관리

   > 프로그램 전체가 하나의 커다란 공간에 연속적으로 할당되어야 함

   - 고정 분할 기법 : 주기억장치가 고정된 파티션으로 분할 (**내부 단편화 발생**)
   - 동적 분할 기법 : 파티션들이 동적 생성되며 자신의 크기와 같은 파티션에 적재 (**외부 단편화 발생**)

   <br>

2. 불연속 메모리 관리

   > 프로그램의 일부가 서로 다른 주소 공간에 할당될 수 있는 기법

   페이지 : 고정 사이즈의 작은 프로세스 조각

   프레임 : 페이지 크기와 같은 주기억장치 메모리 조각

   단편화 : 기억 장치의 빈 공간 or 자료가 여러 조각으로 나뉘는 현상

   세그먼트 : 서로 다른 크기를 가진 논리적 블록이 연속적 공간에 배치되는 것
   <br>

   **고정 크기** : 페이징(Paging)

   **가변 크기** : 세그먼테이션(Segmentation)
   <br>

   - 단순 페이징

     > 각 프로세스는 프레임들과 같은 길이를 가진 균등 페이지로 나뉨
     >
     > 외부 단편화 X
     >
     > 소량의 내부 단편화 존재

   - 단순 세그먼테이션

     > 각 프로세스는 여러 세그먼트들로 나뉨
     >
     > 내부 단편화 X, 메모리 사용 효율 개선, 동적 분할을 통한 오버헤드 감소
     >
     > 외부 단편화 존재

   - 가상 메모리 페이징

     > 단순 페이징과 비교해 프로세스 페이지 전부를 로드시킬 필요X
     >
     > 필요한 페이지가 있으면 나중에 자동으로 불러들어짐
     >
     > 외부 단편화 X
     >
     > 복잡한 메모리 관리로 오버헤드 발생

   - 가상 메모리 세그먼테이션

     > 필요하지 않은 세그먼트들은 로드되지 않음
     >
     > 필요한 세그먼트 있을때 나중에 자동으로 불러들어짐
     >
     > 내부 단편화X
     >
     > 복잡한 메모리 관리로 오버헤드 발생

