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


public class getTCP : MonoBehaviour
{
    private TcpListener listener;
    private Thread receiveThread;
    private int port;

    private float[] px = new float[17];
    private float[] py = new float[17];
    private float[] pz = new float[17];

    public getTCP(int port)
    {
        this.port = port;
    }
    public float[] GetReceivedPX()
    {
        return px;
    }

    public float[] GetReceivedPY()
    {
        return py;
    }

    public float[] GetReceivedPZ()
    {
        return pz;
    }
    public void StartListening()
    {
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    private void ReceiveData()
    {
        try
        {
            listener = new TcpListener(IPAddress.Parse("127.0.0.1"), port);
            listener.Start();
            Byte[] bytes = new Byte[1024];

            while (true)
            {
                using (var client = listener.AcceptTcpClient())
                {
                    using (NetworkStream stream = client.GetStream())
                    {
                        int length;
                        while ((length = stream.Read(bytes, 0, bytes.Length)) != 0)
                        {
                            var incomingData = new byte[length];
                            Array.Copy(bytes, 0, incomingData, 0, length);
                            string clientMessage = Encoding.ASCII.GetString(incomingData);
                            //print(clientMessage);
                            string[] res = clientMessage.Split(' ');
                            for (int i = 0; i < 3; i++)
                            {
                                for (int j = 0; j < 17; j++)
                                {
                                    if (i == 0) px[j] = float.Parse(res[i * 17 + j]);
                                    else if (i == 1) py[j] = float.Parse(res[i * 17 + j]);
                                    else pz[j] = float.Parse(res[i * 17 + j]);
                                }
                            }
                        }
                    }
                }
            }
        }
        catch (Exception e)
        {
            Console.WriteLine(e.ToString());
        }
    }
}

