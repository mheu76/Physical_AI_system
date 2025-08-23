# Security Policy

Physical AI System 프로젝트의 보안 정책입니다.

## 지원되는 버전

현재 보안 업데이트를 받는 Physical AI System 버전은 다음과 같습니다:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## 보안 취약점 보고

Physical AI System에서 보안 취약점을 발견하시면, 다음 절차를 따라 신고해주세요:

### 🚨 긴급 보안 문제

**공개적으로 이슈를 생성하지 마세요.** 대신 다음 중 하나의 방법을 사용해주세요:

1. **이메일**: security@physical-ai.dev
2. **GitHub Security Advisory**: [보안 취약점 신고](https://github.com/your-username/physical-ai-system/security/advisories/new)

### 📧 보고 내용

보안 취약점을 보고할 때는 다음 정보를 포함해주세요:

- **취약점 설명**: 무엇이 잘못되었는지 명확히 설명
- **재현 단계**: 취약점을 재현하는 상세한 단계
- **영향도**: 취약점의 잠재적 영향
- **환경 정보**: OS, Python 버전, 의존성 버전 등
- **가능한 해결책**: 알고 있다면 제안사항

### 🔒 보안 관련 고려사항

Physical AI System은 물리적 로봇과 상호작용하므로 특별한 주의가 필요합니다:

#### 물리적 안전
- **충돌 위험**: 안전 모니터링 시스템 우회 가능성
- **비상 정지**: Emergency stop 기능 무력화
- **하드웨어 제어**: 예기치 않은 하드웨어 동작

#### 데이터 보안
- **센서 데이터**: 개인정보나 민감한 환경 정보 포함 가능
- **학습 데이터**: 행동 패턴이나 사용자 습관 노출
- **설정 파일**: 하드웨어 접근 권한이나 네트워크 정보

#### 네트워크 보안
- **원격 접근**: 무단 로봇 제어 가능성
- **데이터 전송**: 암호화되지 않은 통신
- **API 보안**: 인증/인가 우회

### ⚡ 응답 시간

- **초기 응답**: 48시간 이내
- **취약점 확인**: 7일 이내
- **수정 배포**: 심각도에 따라 1-30일

### 🏆 보안 연구자 인정

보안 취약점을 신고해주신 연구자들을 다음과 같이 인정합니다:

- **Hall of Fame**: SECURITY_CONTRIBUTORS.md 파일에 기록
- **CVE 크레딧**: 해당하는 경우 CVE에 발견자 크레딧
- **감사 인사**: 릴리스 노트와 보안 공지에 감사 표시

## 보안 베스트 프랙티스

Physical AI System을 안전하게 사용하기 위한 권장사항:

### 🛡️ 배포 보안
```bash
# 1. 격리된 환경에서 실행
docker run --rm --network=none physical-ai-system

# 2. 최소 권한으로 실행
docker run --rm --user 1000:1000 physical-ai-system

# 3. 읽기 전용 파일시스템
docker run --rm --read-only physical-ai-system
```

### 🔐 설정 보안
```yaml
# configs/production.yaml
security:
  enable_authentication: true
  encrypt_communications: true
  audit_logging: true
  restrict_hardware_access: true
```

### 📡 네트워크 보안
- 방화벽 설정으로 필요한 포트만 개방
- VPN 또는 전용망을 통한 접근
- TLS 암호화 통신 사용

### 🤖 물리적 보안
- 비상 정지 버튼 항시 접근 가능
- 안전 구역 설정 및 인간 감지
- 정기적인 하드웨어 점검

## 보안 업데이트

보안 업데이트는 다음 채널을 통해 공지됩니다:

- **GitHub Releases**: 보안 패치 릴리스
- **Security Advisories**: 취약점 상세 정보
- **Mailing List**: security-announce@physical-ai.dev

## 규정 준수

Physical AI System은 다음 보안 표준을 준수합니다:

- **ISO 27001**: 정보보안 관리
- **IEC 61508**: 기능안전
- **ROS 2 Security**: 로보틱스 보안 가이드라인

## 연락처

보안 관련 문의사항:

- **보안팀 이메일**: security@physical-ai.dev
- **PGP 키**: [공개키 다운로드](https://physical-ai.dev/pgp-key.asc)
- **보안 정책 문의**: security-policy@physical-ai.dev

---

**보안은 모든 사용자의 책임입니다. 의심스러운 활동을 발견하시면 즉시 신고해주세요.**