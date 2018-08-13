# Git 명령어 모음 by 김경환

## Description

- Git과 Github를 이용하는데 필요한 주요 명령어들을 정리한다. 
- 명령어를 까먹어서 다시 검색해보는 수고를 덜고 git을 처음 쓰는 사람들을 위해 명령어에 대한 간단히 설명한다.
- 자세한 설명과 git에 대한 세부적인 내용들은 다른 자료를 참고하는 것을 추천한다.



### 1. 설정, 초기화 명령어

- `git config --global user.name "<사용자 이름>"`

  `git config --global user.email <사용자 email>`

  - 처음 git 설치 후 전역 사용자명/이메일 설정

- `git init`

  - 현재 디렉토리를 새로운 repository로 초기화(생성)

- `git remote add <name> <remote repository url>`

  - 새로운 원격 저장소 추가

    > `git remote add origin https://github.com/asd/asd.git`

- `git clone <remote repository url>`

  - github repository 복제



### 2. commit 관련 명령어

- `git status`
  - 현재 repository의 상태를 보여줌.
  - 수정된 파일, 새로 추가된 파일, stage에 올라온 파일 등을 보여준다.

- `git add <파일명>`

  - 새로운 파일이나 수정된 파일을 stage에 추가

    > `git add .` : 하위 폴더 내 모든 파일 추가. 삭제된 파일은 추가 안함.
    >
    > `git add -u` : 수정된 파일만 추가.
    >
    > `git add -A` : 모든 파일 추가.

- `git commit`

  - stage에 추가된 내용들을 commit함. 명령어를 실행하면 editor를 통해 commit message를 입력할 수 있다.

    > `git commit -m "<commit message>"` : commit message를 지정해주고 commit 함. commit message가 editor를 거치지 않기 때문에 commit message convention을 검증할 수 없기 때문에 권장하지 않는 방법.
    >
    > `git commit -m "<modified message>" --amend` : 현재 상태를 마지막 commit에 덮어쓰고 message 변경

- `git reset HEAD <filename>`

  - stage에 추가된 파일을 unstage상태로 만듬.

    > `git reset --hard HEAD` : unstage상태로 만들고 실제 파일의 수정사항도 전부 되돌림.

- `git checkout -- <filename>`

  - 파일의 변경사항 원래 상태로 되돌리기.

- `git rm <commit 된 파일 또는 폴더 이름>`

  - commit된 파일 또는 폴더를 commit에서 삭제한다.

    > `git rm -r <commit 된 폴더 이름>` : 폴더와 폴더 내부 파일들 전부 commit에서 삭제한다.



### 3. branch 관련 명령어

- `git branch`

  - 모든 branch를 조회하고 현재 branch를 알려줌.

    > `git branch <branch name>` : 새로운 branch를 생성
    >
    > `git branch -d <branch name>` : 기존 branch 삭제
    >
    > `git branch -r` : remote branch 목록 조회
    >
    > `git branch -a ` : 모든 branch 목록 조회

- `git checkout <branch name>`

  - branch를 변경.

    > `git checkout -b <branch name>` : 새로운 branch를 생성하고 그 branch로 변경.

- `git merge <branch name>`

  - 현재 branch와 지정한 branch를 merge함.
  - [merge에 관한 설명](https://git-scm.com/book/ko/v1/Git-%EB%B8%8C%EB%9E%9C%EC%B9%98-%EB%B8%8C%EB%9E%9C%EC%B9%98%EC%99%80-Merge%EC%9D%98-%EA%B8%B0%EC%B4%88)



### 4. github(remote repository) 관련 명령어

- `git clone <remote repository url>`

  - remote repository 복제해서 local로 가져옴

- `git remote add <name> <remote repository url>`

  - 새로운 remote repository 추가
  - `git remote add origin https://github.com/asd/asd.git`

- `git push <remote repository> <local branch name>`

  - 현재 branch를 remote repository의 branch에 합침. (local -> remote)

    > `git push origin master` : master branch를 origin에 push (defalut)
    >
    > `git push -f origin master` : 강제 push

- `git pull <remote repository> <local branch name>`

  - remote repository의 변경사항을 현재 branch에 합침. (remote -> local)

    > `git pull origin master` : origin을 master branch에 pull (default)

- `git push origin --delete <remote branch name>`

  - remote repository의 branch 삭제

- `git remote prune <remote repository>`

  - remote repository의 branch가 삭제되었을 때 동일한 local branch 제거



### 5. 기타 명령어

#### git log

- `git log`
  - 현재까지의 모든 commit 사항,  날짜 등의 이력을 보여준다.
- `git log -p`
  - 모든 commit에 대한 변경사항까지 보여준다.
- `git log -1`
  - 최근 1개의 이력만 보여준다.
- `git log --oneline`
  - 모든 이력의 commit message, tag를 한 줄씩 보여준다.

#### git diff

- `git diff`
  - 현재 상태와 repository의 차이점(변경사항)을 보여준다.

#### git clean

- `git clean`
  - working directory에서 추적하지 않는 파일을 삭제한다. `.gitignore`에 명시된 파일은 삭제되지 않는다.
- `git clean -f`
  - 강제로 삭제한다.
- `git clean -d`
  - 추적하지 않는 하위 directory도 삭제한다.
- `git clean -x`
  - `.gitignore`에 명시된 파일도 삭제한다.
