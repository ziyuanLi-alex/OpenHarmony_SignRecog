#coding=utf-8

#from BerxelHawkNativeMethods import *
from .BerxelHawkDevice import *
import threading


class DeviceCallback(object):
    def __init__(self):
        self._user_callback = None
        self._real_callback = None

    def setCallback(self, userCallback):
        self._user_callback = userCallback
        if self._user_callback is None:
            self._real_callback = BerxelDeviceStatusCallback()
        else:
            self._real_callback = BerxelDeviceStatusCallback(self._deviceStateCallback)

    def _deviceStateCallback(self, deviceUri, serialNumber, deviceState, userData):
        self._user_callback(deviceUri, serialNumber, deviceState, userData)





class BerxelHawkContext(object):
    #_instance_lock = threading.Lock()
    mDeviceList = []
    mDeviceCount = c_uint32(0)
    mDeviceInfoHandle = deviceInfoHandle()
    deviceCallbackObj = DeviceCallback()

    def __init__(self):
        print("Test init")
        #berxelInit()

    # def __new__(cls, *args, **kwargs):
    #     if not hasattr(cls, '_instance'):
    #         print("new 1")
    #         with BerxelHawkContext._instance_lock:
    #             print("new 2")
    #             if not hasattr(cls, '_instance'):
    #                 print("new 3")
    #                 BerxelHawkContext._instance = super().__new__(cls)
    #                 berxelInit()
    #
    #         return BerxelHawkContext._instance

    # def __del__(self):
    #     print("release list")
    #     berxelReleaseDeviceList(byref(self.mDeviceInfoHandle))
    #     print("destroy2")
    #     berxelDestroy()


    # def destroy(self):
    #     print("destroy")
    #     berxelDestroy()

    def initCamera(self): #用于初始化摄像头系统
        print("init Camera")
        berxelInit()

    def destroyCamera(self): #释放设备列表
        print("destroy Camera")
        berxelReleaseDeviceList(byref(self.mDeviceInfoHandle))
        berxelDestroy()

    def getDeviceList(self): #获取设备
        # deviceInfo_Handle = deviceInfoHandle()
        self.mDeviceList = []
        berxelGetDeviceList(byref(self.mDeviceInfoHandle), byref(self.mDeviceCount))
        if self.mDeviceCount.value < 1:
            return self.mDeviceList
        else:
            for x in range(self.mDeviceCount.value):
                self.mDeviceList.append(self.mDeviceInfoHandle[x])
                print("vid: ", self.mDeviceList[x].vendorId)
                print("addr：", self.mDeviceList[x].deviceAddress)
            #berxelReleaseDeviceList(byref(deviceInfo_Handle))
        return self.mDeviceList


    def setDeviceStausCallBack(self, callback, data): #设置你回调函数？
        BerxelHawkContext.deviceCallbackObj.setCallback(callback)
        return berxelSetDeviceStatusCallback(BerxelHawkContext.deviceCallbackObj._real_callback, data)

    def openDevice(self, deviceinfo): #尝试打开设备返回实例
        device_handle = deviceHandle()

        # if len(BerxelHawkContext.mDeviceList) < 1:
        #     print("not find device")
        #     return None
        #
        # print("openDevice-> addr：", BerxelHawkContext.mDeviceList[0].deviceAddress)
        ret = berxelOpenDeviceByAddr(deviceinfo.deviceAddress, byref(device_handle))
        if(ret == 0):
            return BerxelHawkDevice(device_handle)
        else:
            return None

    def clsoeDevice(self, device): #调用berxelCloseDevice关闭设备
        if device is None:
            return -1
        else:
            return berxelCloseDevice(device._deviceHandle)
