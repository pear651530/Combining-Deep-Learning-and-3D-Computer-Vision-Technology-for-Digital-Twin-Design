using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Net;
using System.Numerics;
using System.Text;
using System.Threading;

using UnityEditor;

using UnityEngine;
using Vector3 = UnityEngine.Vector3;


public class IKSetting : MonoBehaviour
{
    [SerializeField, Range(10, 120)] float FrameRate=30;
    public List<Transform> BoneList = new List<Transform>();
    [SerializeField] string Data_Path;
    [SerializeField] string File_Name;
    [SerializeField] int Data_Size;
    public Vector3 CameraP = new Vector3(0.27f, 0.75f, 5.39f);
    GameObject FullbodyIK;
    private GameObject leftthumb;
    Vector3[] points = new Vector3[17];
    Vector3[] NormalizeBone = new Vector3[12];
    float[] BoneDistance = new float[12];
    float Timer;
    int[, ] joints = new int[, ] { { 0, 1 }, { 1, 2 }, { 2, 3 }, { 0, 4 }, { 4, 5 }, { 5, 6 }, { 0, 7 }, { 7, 8 }, { 8, 9 }, { 9, 10 }, { 8, 11 }, { 11, 12 }, { 12, 13 }, { 8, 14 }, { 14, 15 }, { 15, 16 } };
    int[, ] BoneJoint = new int[, ] { { 0, 2 }, { 2, 3 }, { 0, 5 }, { 5, 6 }, { 0, 9 }, { 9, 10 }, { 9, 11 }, { 11, 12 }, { 12, 13 }, { 9, 14 }, { 14, 15 }, { 15, 16 } };
    int[, ] NormalizeJoint = new int[, ] { { 0, 1 }, { 1, 2 }, { 0, 3 }, { 3, 4 }, { 0, 5 }, { 5, 6 }, { 5, 7 }, { 7, 8 }, { 8, 9 }, { 5, 10 }, { 10, 11 }, { 11, 12 } };
    int NowFrame = 0;
    private getTCP aaa;
    public float[] px = new float[17];
    public float[] py = new float[17];
    public float[] pz = new float[17];
    // 查找所有命名为"fullbodyik"的对象

    void Start()
    {
        aaa = new getTCP(5066);
        print('a');
        aaa.StartListening();
        PointUpdate();
    }
    void Update()
    {
        px = aaa.GetReceivedPX();
        py = aaa.GetReceivedPY();
        pz = aaa.GetReceivedPZ();
        //print(px);
        Timer += Time.deltaTime;
        if (Timer > (1 / FrameRate))
        {
            Timer = 0;
            PointUpdate();
        }
        //print(FullbodyIK);
        if (!FullbodyIK)
        {
            IKFind();
        }
        else
        {
            IKSet();
        }
    }
    void PointUpdate()
    {
        //StreamReader fi = null;
        
        bool flag = false;
        for (int i = 0; i < 17; i++)
        {
            if (px[i] != 0 || py[i] != 0 || pz[i] != 0)
            {
                flag = true;
                break;
            }
        }
        if (flag)
        {
            for (int i = 0; i < 17; i++)
            {
                points[i] = new Vector3(px[i], py[i], -pz[i]);  //get 17 joint points coordination 
            }
            for (int i = 0; i < 12; i++)
            {
                NormalizeBone[i] = (points[BoneJoint[i, 1]] - points[BoneJoint[i, 0]]).normalized;  //Normalized skeleton length
            }
        }
        else
        {
            //Debug.Log("All Data 0");
        }
    }
    void IKFind()
    {
        FullbodyIK = GameObject.Find("FullBodyIK");
        if (FullbodyIK)
        {
            for (int i = 0; i < Enum.GetNames(typeof(OpenPoseRef)).Length; i++)
            {
                //print('a');
                //print(Enum.GetName(typeof(OpenPoseRef), i));
                Transform obj = GameObject.Find(Enum.GetName(typeof(OpenPoseRef), i)).transform;
                if (obj)
                {
                    BoneList.Add(obj);
                }
            }
            for (int i = 0; i < Enum.GetNames(typeof(NormalizeBoneRef)).Length; i++)
            {
                BoneDistance[i] = Vector3.Distance(BoneList[NormalizeJoint[i, 0]].position, BoneList[NormalizeJoint[i, 1]].position);
            }
        }
    }
    void IKSet()
    {
        if (Math.Abs(points[0].x) < 1000 && Math.Abs(points[0].y) < 1000 && Math.Abs(points[0].z) < 1000)
        {
            BoneList[0].position = Vector3.Lerp(BoneList[0].position, points[0] + Vector3.up * 0.8f, 0.1f); //骨骼髋节点的位置
            FullbodyIK.transform.position = Vector3.Lerp(FullbodyIK.transform.position, points[0], 0.01f); //全身骨架的根节点位置
            Vector3 hipRot = (NormalizeBone[0] + NormalizeBone[2] + NormalizeBone[4]).normalized; //计算实时髋节点的方向
            FullbodyIK.transform.forward = Vector3.Lerp(FullbodyIK.transform.forward, new Vector3(hipRot.x, 0, hipRot.z), 0.1f); // 整体骨架的Z轴方向
        }
        for (int i = 0; i < 12; i++)
        {
            BoneList[NormalizeJoint[i, 1]].position = Vector3.Lerp(
                BoneList[NormalizeJoint[i, 1]].position,
                BoneList[NormalizeJoint[i, 0]].position + BoneDistance[i] * NormalizeBone[i], 0.05f  // 骨骼长度*方向向量
            );
            DrawLine(BoneList[NormalizeJoint[i, 0]].position + Vector3.right, BoneList[NormalizeJoint[i, 1]].position + Vector3.right, Color.red);
        }
       for (int i = 0; i < joints.Length / 2 ; i++)
       {
           DrawLine(points[joints[i, 0]] * 0.001f + new Vector3(-1, 0.8f, 0), points[joints[i, 1]] * 0.001f + new Vector3(-1, 0.8f, 0), Color.blue);
       }
    }
    // 画12根骨架位置
    void DrawLine(Vector3 s, Vector3 e, Color c)
    {
        Debug.DrawLine(s, e, c);
    }
}
enum OpenPoseRef
{
    Hips,

    RightKnee,
    RightFoot,
    LeftKnee,
    LeftFoot,
    Neck,
    Head,

    LeftArm,
    LeftElbow,
    LeftWrist,
    RightArm,
    RightElbow,
    RightWrist,
};
enum NormalizeBoneRef
{
    Hip2LeftKnee,
    LeftKnee2LeftFoot,
    Hip2RightKnee,
    RightKnee2RightFoot,
    Hip2Neck,
    Neck2Head,
    Neck2RightArm,
    RightArm2RightElbow,
    RightElbow2RightWrist,
    Neck2LeftArm,
    LeftArm2LeftElbow,
    LeftElbow2LeftWrist
};
