from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass

class Role(Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    EMPLOYEE = "employee"

@dataclass
class User:
    username: str
    password: str
    role: Role
    allowed_documents: List[str]

class AuthManager:
    def __init__(self):
        # En un entorno real, esto debería estar en una base de datos
        self.users: Dict[str, User] = {
            "admin": User("admin", "admin123", Role.ADMIN, ["*"]),
            "manager": User("manager", "manager123", Role.MANAGER, ["manual_operaciones.pdf"]),
            "employee": User("employee", "employee123", Role.EMPLOYEE, ["preguntas_frecuentes.txt"])
        }
    
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Autentica a un usuario y retorna su información si es válido."""
        user = self.users.get(username)
        if user and user.password == password:
            return user
        return None
    
    def can_access_document(self, username: str, document_name: str) -> bool:
        """Verifica si un usuario tiene acceso a un documento específico."""
        user = self.users.get(username)
        if not user:
            return False
        
        if "*" in user.allowed_documents:
            return True
        
        return document_name in user.allowed_documents 