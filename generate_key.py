#!/usr/bin/env python3
"""Generate SSH keypair for GitHub deploy key."""

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

def main():
    # Generate key pair
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    # Save private key (PKCS8 PEM)
    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    with open('github_deploy_key', 'wb') as f:
        f.write(priv_pem)

    # Save public key (OpenSSH format)
    pub_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH
    )
    with open('github_deploy_key.pub', 'wb') as f:
        f.write(pub_bytes + b' subtitle-translator-bot')

    print("=" * 60)
    print("SSH KEYPAIR GENERATED")
    print("=" * 60)
    print()
    print("PASTE THIS PUBLIC KEY TO GITHUB DEPLOY KEYS:")
    print()
    pub_str = pub_bytes.decode() + " subtitle-translator-bot"
    print(pub_str)
    print()
    print("=" * 60)
    print("Private key saved to: github_deploy_key")
    print("Public key saved to: github_deploy_key.pub")
    print("=" * 60)


if __name__ == "__main__":
    main()
