int iJIT_NotifyEvent(unsigned int event_type, void *event_specific_data) {
    (void)event_type;
    (void)event_specific_data;
    return 0;
}

int iJIT_IsProfilingActive(void) {
    return 0;
}

unsigned int iJIT_GetNewMethodID(void) {
    static unsigned int method_id = 1;
    return method_id++;
}
