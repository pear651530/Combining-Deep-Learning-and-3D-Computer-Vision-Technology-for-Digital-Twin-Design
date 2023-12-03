using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraP : MonoBehaviour
{
    public Vector3 newPosition = new Vector3(0.01f, 0.96f, 3.15f); // 新的攝影機位置
    public Vector3 newRotation = new Vector3(0f, -180f, 0f);
    private void Update()
    {
        // 取得主攝影機的Transform組件
        Transform cameraTransform = Camera.main.transform;

        // 設置攝影機的新位置
        cameraTransform.position = newPosition;

        cameraTransform.rotation = Quaternion.Euler(newRotation);
    }
}
